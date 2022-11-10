#ifndef HYBRID_PART_H_INCLUDED
#define HYBRID_PART_H_INCLUDED

#include <Eigen/Core>
#include <vector>
#include <types.h>

namespace pspy {

struct HybridPart {
	HybridPart(
		const std::string& path, 
		const int N, 
		const int N_ref, 
		const bool normalize,
		const double sorted_frac = 0.5);
	void ApplyTransform(const Eigen::RowVector3d& translation, const double scale);

	// Global Bounding Box
	Eigen::MatrixXd bounding_box;

	// Node Data
	std::vector<SurfaceFunction> face_surfaces;
	Eigen::MatrixXd face_surface_parameters;
	Eigen::VectorXi face_surface_flipped;

	std::vector<LoopType> loop_types;

	Eigen::VectorXd loop_length;

	std::vector<CurveFunction> edge_curves;
	Eigen::MatrixXd edge_curve_parameters;
	Eigen::VectorXi edge_curve_flipped;

	Eigen::VectorXd edge_length;

	Eigen::MatrixXd vertex_positions;

	// Graph Topology
	Eigen::MatrixXi face_to_face;
	Eigen::MatrixXi face_to_loop;
	Eigen::MatrixXi loop_to_edge;
	Eigen::MatrixXi face_to_edge;
	Eigen::MatrixXi edge_to_vertex;
	Eigen::MatrixXi loop_to_vertex;

	// If a vertex is start, end, or neither relative to an edge
	std::vector<bool> edge_to_vertex_is_start;
	std::vector<bool> loop_to_edge_flipped;
	std::vector<bool> face_to_edge_flipped;

	// Surface and Curve Samples
	std::vector<Eigen::MatrixXd> surface_bounds; // N_face x 2 x 2
	std::vector<Eigen::MatrixXd> surface_coords; // N_face x N x 2
	std::vector<Eigen::MatrixXd> surface_samples; // N_face x N x 7
	std::vector<Eigen::Vector2d> curve_bounds; // N_edge x 2
	std::vector<Eigen::MatrixXd> curve_samples; // N_edge x ceil(sqrt(N)) x 6

	Eigen::RowVector3d translation;
	double scale;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::VectorXi FtoT;
	Eigen::MatrixXi EtoT;
	Eigen::VectorXi VtoT;

	// Kernel-Computed Features
	Eigen::MatrixXd F_k_feats; // surface_area, circumference, bounding_box, na_bounding_box, center_of_gravity, moment_of_inertia
	Eigen::MatrixXd E_k_feats; // t_range, start, end, mid_point, length, bounding_box, na_bounding_box, center_of_gravity, moment_of_inertia

	bool valid;
};

}

#endif // !HYBRID_PART_H_INCLUDED