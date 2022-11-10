#include "hybrid_part.h"

#include <body.h>

namespace pspy {

	const double SURF_ERROR_FRAC = 0.004;

HybridPart::HybridPart(
	const std::string& path, 
	const int N,
	const int N_ref,
	const bool normalize,
	double sorted_frac)
{
	valid = true;

	auto bodies = read_file(path);
	if (bodies.size() != 1) {
		valid = false;
		return;
	}

	Body& body = *bodies[0];

	bounding_box = body.GetBoundingBox();

	auto topology = body.GetTopology();

	auto max_dim = (bounding_box.row(1) - bounding_box.row(0)).maxCoeff();
	double max_surf_error = max_dim * SURF_ERROR_FRAC;

	body.Tesselate(V, F, FtoT, EtoT, VtoT, true, max_surf_error);

	const int n_faces = topology.faces.size();
	const int n_edges = topology.edges.size();
	const int n_loops = topology.loops.size();

	// Get Samples
	const int N_curve = (int)(ceil(sqrt(N)));

	surface_bounds.resize(n_faces);
	surface_coords.resize(n_faces);
	surface_samples.resize(n_faces);
	for (int i = 0; i < n_faces; ++i) {
		if (!topology.faces[i]->sample_surface(N_ref, N, surface_bounds[i], surface_coords[i], surface_samples[i], true, sorted_frac)) {
			valid = false;
			return;
		}
	}

	curve_bounds.resize(n_edges);
	curve_samples.resize(n_edges);
	for (int i = 0; i < n_edges; ++i) {
		if (!topology.edges[i]->sample_curve(N_curve, curve_bounds[i], curve_samples[i])) {
			valid = false;
			return;
		}
	}

	// Get kernel features
	// We generally do not want to use these, but we do want to see if we can get better results with them
	// Eigen::MatrixXd F_k_feats; surface_area, circumference, bounding_box, na_bounding_box, center_of_gravity, moment_of_inertia
	// Eigen::MatrixXd E_k_feats; t_range, start, end, mid_point, length, bounding_box, na_bounding_box, center_of_gravity, moment_of_inertia

	// Transformation Properties:
	// Scale: 0, 1, 2, 
	F_k_feats.resize(n_faces, 26);
	for (int i = 0; i < n_faces; ++i) {
		// 26 wide, 36 if we use the rest of the na_bb
		double surface_area = topology.faces[i]->surface_area; // 1 -- scales as square
		F_k_feats(i,0) = surface_area;
		double circumference = topology.faces[i]->circumference; // 1 -- scales
		F_k_feats(i,1) = circumference;
		auto bounding_box = topology.faces[i]->bounding_box; // 6 (2x3) -- scales, translates
		for (int r = 0; r < 2; ++r) {
			for (int c = 0; c < 3; ++c) {
				F_k_feats(i,2+(2*r+c)) = bounding_box(r,c);
			}
		}
		auto na_bounding_box = topology.faces[i]->na_bounding_box; // 6 (2x3) -- scales
		for (int r = 0; r < 2; ++r) {
			for (int c = 0; c < 3; ++c) {
				F_k_feats(i,8+(2*r+c)) = na_bounding_box(r,c);
			}
		}
		auto na_bb_center = topology.edges[i]->na_bb_center; // ignored for now
		auto na_bb_x = topology.edges[i]->na_bb_x; // ignored for now
		auto na_bb_z = topology.edges[i]->na_bb_z; // ignored for now
		auto center_of_gravity = topology.faces[i]->center_of_gravity; // 3 -- scales, translates
		for (int j = 0; j < 3; ++j) {
			F_k_feats(i,14+j) = center_of_gravity(j);
		}
		auto moment_of_inertia = topology.faces[i]->moment_of_inertia; // 9 (3x3) - relative to center of gravity, scales at 5th power
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				F_k_feats(i,17+(2*r+c)) = moment_of_inertia(r,c);
			}
		}
	}

	E_k_feats.resize(n_edges, 36);
	for (int i = 0; i < n_edges; ++i) {
		// 36 wide, 45 if we use the rest of the na_bb
		auto t_start = topology.edges[i]->t_start; // 1
		E_k_feats(0) = t_start;
		auto t_end = topology.edges[i]->t_end; // 1
		E_k_feats(1) = t_end;
		auto start = topology.edges[i]->start; // 3 -- scales, translates
		for (int j = 0; j < 3; ++j) {
			E_k_feats(i,2+j) = start(j);
		}
		auto end = topology.edges[i]->end; // 3 -- scales, translates
		for (int j = 0; j < 3; ++j) {
			E_k_feats(i,5+j) = end(j);
		}
		auto mid_point = topology.edges[i]->mid_point; // 3 -- scales, translates
		for (int j = 0; j < 3; ++j) {
			E_k_feats(i,8+j) = mid_point(j);
		}
		auto length = topology.edges[i]->length; // 1 -- scales
		E_k_feats(11) = length;
		auto bounding_box = topology.edges[i]->bounding_box; // 6 (2x3) (min, max) -- scales, translates
		for (int r = 0; r < 2; ++r) {
			for (int c = 0; c < 3; ++c) {
				E_k_feats(i,12+(2*r+c)) = bounding_box(r,c);
			}
		}
		auto na_bounding_box = topology.edges[i]->na_bounding_box; // 6 (2x3) -- scales
		for (int r = 0; r < 2; ++r) {
			for (int c = 0; c < 3; ++c) {
				E_k_feats(i,18+(2*r+c)) = na_bounding_box(r,c);
			}
		}
		auto na_bb_center = topology.edges[i]->na_bb_center; // ignored since not used currently in automate
		auto na_bb_box_x = topology.edges[i]->na_bb_x; // ignored since not used currenty in automate
		auto na_bb_z = topology.edges[i]->na_bb_z; // ignored since not used currently in automate
		auto center_of_gravity = topology.edges[i]->center_of_gravity; // 3 -- scales, translates
		for (int j = 0; j < 3; ++j) {
			E_k_feats(i,24+j) = center_of_gravity(j);
		}
		auto moment_of_inertia = topology.edges[i]->moment_of_inertia; // 9 (3x3) - relative to center of gravity -- scales at 5th power
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				E_k_feats(i,27+(2*r+c)) = moment_of_inertia(r,c);
			}
		}
	}

	// Setup Node Data
	// Surface Data Format:
	// [type] | [origin] [normal] [axis] [ref_dir] [radius] [minor_radius] [semi-angle]
	// [    ] | [0:3]    [3:6]    [6:9]  [9:12]    [12]     [13]           [14]
	// Another thought (maybe just on the learning / python side) is to pre-transform these
	// using tanh or something similar to constrain the range in a reversible way. The big
	// problem with this is then that the majority of the representable space is taken up
	// by uncommon values
	// Default values:
	// plane: normal = N, axis = 0, ref_dir = x, radius = minor_radius = semi-angle = 0
	// cylinder: normal = 0, axis= a, ref_dir = x, radius = minor_radius = r, semi_angle = 0
	// cone: axis, ref_dir, radius = minor_radius, semi-angle = a
	// sphere: axis, ref_dir, radius = minor_radius, semi-angle = 0
	// torus: axis, ref_dir, radius
	// Idea: let the unused direction be the cross product (y) direction
	const int SURF_PARAM_WIDTH = 15;
	face_surfaces.resize(n_faces);
	face_surface_parameters.resize(n_faces, SURF_PARAM_WIDTH);
	face_surface_parameters.setZero();
	face_surface_flipped.resize(n_faces);
	for (int i = 0; i < n_faces; ++i) {
		face_surfaces[i] = topology.faces[i]->function;
		switch (face_surfaces[i]) {
			case SurfaceFunction::PLANE:
				for (int j = 0; j < 3; ++j) {
					face_surface_parameters(i,0+j) = topology.faces[i]->parameters[0+j]; // origin [0:3]
					face_surface_parameters(i,3+j) = topology.faces[i]->parameters[3+j]; // normal [3:6]
					face_surface_parameters(i,9+j) = topology.faces[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				break;
			case SurfaceFunction::CYLINDER:
				for (int j = 0; j < 3; ++j) {
					face_surface_parameters(i,0+j) = topology.faces[i]->parameters[0+j]; // origin [0:3]
					face_surface_parameters(i,6+j) = topology.faces[i]->parameters[3+j]; // axis [6:9]
					face_surface_parameters(i,9+j) = topology.faces[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				face_surface_parameters(i,12) = topology.faces[i]->parameters[9]; // radius [12]
				break;
			case SurfaceFunction::CONE:
				for (int j = 0; j < 3; ++j) {
					face_surface_parameters(i,0+j) = topology.faces[i]->parameters[0+j]; // origin [0:3]
					face_surface_parameters(i,6+j) = topology.faces[i]->parameters[3+j]; // axis [6:9]
					face_surface_parameters(i,9+j) = topology.faces[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				face_surface_parameters(i,12) = topology.faces[i]->parameters[9]; // radius [12]
				face_surface_parameters(i,14) = topology.faces[i]->parameters[10]; // semi-angle [14]
				break;
			case SurfaceFunction::TORUS:
				for (int j = 0; j < 3; ++j) {
					face_surface_parameters(i,0+j) = topology.faces[i]->parameters[0+j]; // origin [0:3]
					face_surface_parameters(i,6+j) = topology.faces[i]->parameters[3+j]; // axis [6:9]
					face_surface_parameters(i,9+j) = topology.faces[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				face_surface_parameters(i,12) = topology.faces[i]->parameters[9]; // radius [12]
				face_surface_parameters(i,13) = topology.faces[i]->parameters[10]; // minor-radius [13]
				break;
			case SurfaceFunction::SPHERE:
				for (int j = 0; j < 3; ++j) {
					face_surface_parameters(i,0+j) = topology.faces[i]->parameters[0+j]; // origin [0:3]
					face_surface_parameters(i,6+j) = topology.faces[i]->parameters[3+j]; // axis [6:9]
					face_surface_parameters(i,9+j) = topology.faces[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				face_surface_parameters(i,12) = topology.faces[i]->parameters[9]; // radius [12]
				break;
		}

		face_surface_flipped[i] = (int)(!topology.faces[i]->orientation);
	}

	loop_types.resize(n_loops);
	for (int i = 0; i < n_loops; ++i) {
		loop_types[i] = topology.loops[i]->_type;
	}

	// Cuve Parameter Format:
	// [fn] | [origin] [direction] [axis] [x_dir] [radius] [minor radius]
	// [  ] | [0:3   ] [3:6]       [6:9]  [9:12 ] [12]     [13]
	const int CURVE_PARAM_WIDTH = 14;
	edge_curves.resize(n_edges);
	edge_curve_parameters.resize(n_edges, CURVE_PARAM_WIDTH);
	edge_curve_parameters.setZero();
	edge_length.resize(n_edges);
	edge_curve_flipped.resize(n_edges);
	for (int i = 0; i < n_edges; ++i) {
		edge_curves[i] = topology.edges[i]->function;

		switch (edge_curves[i]) {
			case CurveFunction::LINE:
				for (int j = 0; j < 3; ++j) {
					edge_curve_parameters(i,0+j) = topology.edges[i]->parameters[0+j]; // origin [0:3]
					edge_curve_parameters(i,3+j) = topology.edges[i]->parameters[3+j]; // dir [3:6]
				}
				break;
			case CurveFunction::CIRCLE:
				for (int j = 0; j < 3; ++j) {
					edge_curve_parameters(i,0+j) = topology.edges[i]->parameters[0+j]; // origin [0:3]
					edge_curve_parameters(i,6+j) = topology.edges[i]->parameters[3+j]; // axis [6:9]
					edge_curve_parameters(i,9+j) = topology.edges[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				edge_curve_parameters(i,12) = topology.edges[i]->parameters[9]; // radius [12]
				break;
			case CurveFunction::ELLIPSE:
				for (int j = 0; j < 3; ++j) {
					edge_curve_parameters(i,0+j) = topology.edges[i]->parameters[0+j]; // origin [0:3]
					edge_curve_parameters(i,6+j) = topology.edges[i]->parameters[3+j]; // axis [6:9]
					edge_curve_parameters(i,9+j) = topology.edges[i]->parameters[6+j]; // ref_dir (x) [9:12]
				}
				edge_curve_parameters(i,12) = topology.edges[i]->parameters[9]; // radius [12]
				edge_curve_parameters(i,13) = topology.edges[i]->parameters[10]; // minor-radius [13]
				break;
		}

		edge_length[i] = topology.edges[i]->length;
		edge_curve_flipped[i] = topology.edges[i]->_is_reversed;
	}

	vertex_positions.resize(topology.vertices.size(), 3);
	for (int i = 0; i < topology.vertices.size(); ++i) {
		vertex_positions.row(i) = topology.vertices[i]->position;
	}

	// Setup Toploogy Structures
	std::map<int, int> loop_to_face;
	face_to_loop.resize(2, topology.face_to_loop.size());
	for (int i = 0; i < topology.face_to_loop.size(); ++i) {
		face_to_loop(0, i) = topology.face_to_loop[i]._parent;
		face_to_loop(1, i) = topology.face_to_loop[i]._child;
		loop_to_face[face_to_loop(1,i)] = face_to_loop(0,i);
	}

	loop_to_edge.resize(2, topology.loop_to_edge.size());
	loop_to_edge_flipped.resize(topology.loop_to_edge.size());
	face_to_edge.resize(2, topology.loop_to_edge.size());
	face_to_edge_flipped.resize(topology.loop_to_edge.size());
	loop_length.resize(n_loops);
	loop_length.setZero();
	for (int i = 0; i < topology.loop_to_edge.size(); ++i) {
		int loop_idx = topology.loop_to_edge[i]._parent;
		int edge_idx = topology.loop_to_edge[i]._child;
		loop_to_edge(0, i) = loop_idx;
		loop_to_edge(1, i) = edge_idx;
		loop_to_edge_flipped[i] = (topology.loop_to_edge[i]._sense == TopoRelationSense::Negative);
		loop_length(loop_idx) += edge_length(edge_idx);
		face_to_edge(0,i) = loop_to_face[loop_idx];
		face_to_edge(1,i) = edge_idx;
		face_to_edge_flipped[i] = loop_to_edge_flipped[i];
	}

	

	edge_to_vertex.resize(2, topology.edge_to_vertex.size());
	edge_to_vertex_is_start.resize(topology.edge_to_vertex.size());
	for (int i = 0; i < topology.edge_to_vertex.size(); ++i) {
		int edge_idx = topology.edge_to_vertex[i]._parent;
		int vert_idx = topology.edge_to_vertex[i]._child;
		edge_to_vertex(0, i) = edge_idx;
		edge_to_vertex(1, i) = vert_idx;

		// Very few ~7k edges have only 1 vertex (likely a bug or invalid data?)
		// Parasolid docs indicate it can happen, not sure how though
		// For now, just decide if it is a start vertex or not

		// If the edge is flipped, we need to reverse the start and end
		bool flipped = topology.edges[edge_idx]->_is_reversed;

		auto s = curve_samples[edge_idx].row(flipped ? N_curve -1 : 0);
		auto e = curve_samples[edge_idx].row(flipped ? 0 : N_curve - 1);
		Eigen::Vector3d start(s(0), s(1), s(2));
		Eigen::Vector3d end(e(0), e(1), e(2));
		Eigen::Vector3d pos = topology.vertices[vert_idx]->position;
		double start_dist = (start - pos).norm();
		double end_dist = (end - pos).norm();
		edge_to_vertex_is_start[i] = (start_dist <= end_dist);

	}

	face_to_face.resize(3, topology.face_to_face.size());
	for (int i = 0; i < topology.face_to_face.size(); ++i) {
		face_to_face(0, i) = std::get<0>(topology.face_to_face[i]);
		face_to_face(1, i) = std::get<1>(topology.face_to_face[i]);
		face_to_face(2, i) = std::get<2>(topology.face_to_face[i]);
	}

	loop_to_vertex.resize(2, topology.loop_to_vertex.size());
	for (int i = 0; i < topology.loop_to_vertex.size(); ++i) {
		loop_to_vertex(0, i) = topology.loop_to_vertex[i]._parent;
		loop_to_vertex(1, i) = topology.loop_to_vertex[i]._child;
	}

	translation = Eigen::RowVector3d(0.0, 0.0, 0.0);
	scale = 1.0;
	if (normalize && valid) {
		// Get Scaling Transformation
		// Translate and scale to fit in [-1,1]^3
		Eigen::RowVector3d min_corner = bounding_box.row(0);
		Eigen::RowVector3d max_corner = bounding_box.row(1);
		Eigen::RowVector3d diag = max_corner - min_corner;
		Eigen::RowVector3d loc_translation = - (max_corner + min_corner) / 2;
		double max_dim = diag.maxCoeff();
		if (max_dim <= 0) {
			valid = false;
			return;
		}
		double loc_scale = 2 / diag.maxCoeff();
		ApplyTransform(loc_translation, loc_scale);
	}
}

void HybridPart::ApplyTransform(
	const Eigen::RowVector3d& translation, 
	const double scale
)
{
	this->translation += translation;
	this->scale *= scale;

	const int n_faces = face_surfaces.size();
	const int n_edges = edge_curves.size();

	// Transform Mesh
	V.rowwise() += translation;
	V *= scale;

	// Apply to surface parameters
	// [origin] [normal] [axis] [ref_dir] [radius] [minor_radius] [semi-angle]
	// [0:3]    [3:6]    [6:9]  [9:12]    [12]     [13]           [14]
	//  t+s      ---      ---    ---       s        s              ---

	face_surface_parameters.block(0, 0, n_faces, 3).rowwise() += translation;
	face_surface_parameters.block(0, 0, n_faces, 3) *= scale;
	face_surface_parameters.block(0, 12, n_faces, 2) *= scale;


	// Apply to edge parameters
	// [origin] [direction] [axis] [x_dir] [radius] [minor radius]
	// [0:3   ] [3:6]       [6:9]  [9:12 ] [12]     [13]
	//  t+s      ---         ---    ---     s        s

	edge_curve_parameters.block(0, 0, n_edges, 3).rowwise() += translation;
	edge_curve_parameters.block(0, 0, n_edges, 3) *= scale;
	edge_curve_parameters.block(0, 12, n_edges, 2) *= scale;

	// Apply to vertex positions
	vertex_positions.rowwise() += translation;
	vertex_positions *= scale;

	// Apply to samples
	// cols 0-3 translate and scale (x,y,z)
	for (int i = 0; i < n_faces; ++i) {
		const int N = surface_samples[i].rows();
		surface_samples[i].block(0, 0, N, 3).rowwise() += translation;
		surface_samples[i].block(0, 0, N, 3) *= scale;
	}
	for (int i = 0; i < n_edges; ++i) {
		const int N = curve_samples[i].rows();
		curve_samples[i].block(0, 0, N, 3).rowwise() += translation;
		curve_samples[i].block(0, 0, N, 3) *= scale;
	}

	// Apply to bounds
	// Only scale non-periodic dimensions
	for (int i = 0; i < n_faces; ++i) {
		switch (face_surfaces[i]) {
		case SurfaceFunction::PLANE:
			surface_bounds[i] *= scale; // scale u and v
			break;
		case SurfaceFunction::CYLINDER:
			surface_bounds[i].col(1) *= scale; // scale v
			break;
		case SurfaceFunction::CONE:
			surface_bounds[i].col(1) *= scale; // scale v
			break;
		case SurfaceFunction::SPHERE:
			break;
		case SurfaceFunction::TORUS:
			break;
		}
	}

	for (int i = 0; i < n_edges; ++i) {
		if (edge_curves[i] == CurveFunction::LINE) {
			curve_bounds[i] *= scale;
		}
	}

	// Apply to edge and loop lengths
	edge_length *= scale;
	loop_length *= scale;

	// Apply to bounding box
	bounding_box.rowwise() += translation;
	bounding_box *= scale;

	// Apply to the Kernel Features
	// this is a tricky operation since different features scale differently
	F_k_feats.block(0,0,n_faces,1) *= scale*scale; // surface area

	F_k_feats.block(0,1,n_faces,1) *= scale; // circumference
	
	F_k_feats.block(0,2,n_faces,1).array() += translation(0); // bb_min_x
	F_k_feats.block(0,3,n_faces,1).array() += translation(1); // bb_min_y
	F_k_feats.block(0,4,n_faces,1).array() += translation(2); // bb_min_y
	F_k_feats.block(0,5,n_faces,1).array() += translation(0); // bb_max_x
	F_k_feats.block(0,6,n_faces,1).array() += translation(1); // bb_max_y
	F_k_feats.block(0,7,n_faces,1).array() += translation(2); // bb_max_y
	F_k_feats.block(0,2,n_faces,6) *= scale; // bb
	
	F_k_feats.block(0,8,n_faces,6) *= scale; // na_bb

	F_k_feats.block(0,14,n_faces,3).rowwise() += translation; // c_of_g
	F_k_feats.block(0,14,n_faces,3) *= scale; // c_of_g
	F_k_feats.block(0,17,n_faces,9) *= scale*scale*scale*scale*scale; // m_o_i

	E_k_feats.block(0,2,n_edges,3).rowwise() += translation; // start 
	E_k_feats.block(0,2,n_edges,3) *= scale; // start

	E_k_feats.block(0,5,n_edges,3).rowwise() += translation; // end 
	E_k_feats.block(0,5,n_edges,3) *= scale; // end

	E_k_feats.block(0,8,n_edges,3).rowwise() += translation; // mid_point 
	E_k_feats.block(0,8,n_edges,3) *= scale; // mid_point

	E_k_feats.block(0,11,n_edges,1) *= scale; // length

	E_k_feats.block(0,12,n_edges,1).array() += translation(0); // bb_min_x
	E_k_feats.block(0,13,n_edges,1).array() += translation(1); // bb_min_y
	E_k_feats.block(0,14,n_edges,1).array() += translation(2); // bb_min_y
	E_k_feats.block(0,15,n_edges,1).array() += translation(0); // bb_max_x
	E_k_feats.block(0,16,n_edges,1).array() += translation(1); // bb_max_y
	E_k_feats.block(0,17,n_edges,1).array() += translation(2); // bb_max_y
	E_k_feats.block(0,12,n_edges,6) *= scale; // bb

	E_k_feats.block(0,18,n_edges,6) *= scale; // na_bb

	E_k_feats.block(0,24,n_edges,3).rowwise() += translation; // c_of_g
	E_k_feats.block(0,24,n_edges,3) *= scale; // c_of_g

	E_k_feats.block(0,27,n_edges,9) *= scale*scale*scale*scale*scale; // m_o_i

}

}