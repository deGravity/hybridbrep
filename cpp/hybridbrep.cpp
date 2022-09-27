#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "implicit_part.h"
#include "hybrid_part.h"

namespace py = pybind11;
using namespace pspy;

PYBIND11_MODULE(hybridbrep_cpp, m) {
	// implicit_part.h
	py::class_<ImplicitPart>(m, "ImplicitPart")
		.def(py::init<const std::string&, const int, const int, const bool>())
		.def("ApplyTransform", &ImplicitPart::ApplyTransform)
		.def_readwrite("bounding_box", &ImplicitPart::bounding_box)
		.def_readwrite("face_surfaces", &ImplicitPart::face_surfaces)
		.def_readwrite("face_surface_parameters", &ImplicitPart::face_surface_parameters)
		.def_readwrite("face_surface_flipped", &ImplicitPart::face_surface_flipped)
		.def_readwrite("loop_types", &ImplicitPart::loop_types)
		.def_readwrite("loop_length", &ImplicitPart::loop_length)
		.def_readwrite("edge_curves", &ImplicitPart::edge_curves)
		.def_readwrite("edge_curve_parameters", &ImplicitPart::edge_curve_parameters)
		.def_readwrite("edge_curve_flipped", &ImplicitPart::edge_curve_flipped)
		.def_readwrite("edge_length", &ImplicitPart::edge_length)
		.def_readwrite("vertex_positions", &ImplicitPart::vertex_positions)
		.def_readwrite("face_to_face", &ImplicitPart::face_to_face)
		.def_readwrite("face_to_loop", &ImplicitPart::face_to_loop)
		.def_readwrite("loop_to_edge", &ImplicitPart::loop_to_edge)
		.def_readwrite("edge_to_vertex", &ImplicitPart::edge_to_vertex)
		.def_readwrite("loop_to_vertex", &ImplicitPart::loop_to_vertex)
		.def_readwrite("ordered_loop_edge", &ImplicitPart::ordered_loop_edge)
		.def_readwrite("ordered_loop_flipped", &ImplicitPart::ordered_loop_flipped)
		.def_readwrite("edge_to_vertex_is_start", &ImplicitPart::edge_to_vertex_is_start)
		.def_readwrite("loop_to_edge_flipped", &ImplicitPart::loop_to_edge_flipped)
		.def_readwrite("surface_bounds", &ImplicitPart::surface_bounds)
		.def_readwrite("surface_coords", &ImplicitPart::surface_coords)
		.def_readwrite("surface_samples", &ImplicitPart::surface_samples)
		.def_readwrite("curve_bounds", &ImplicitPart::curve_bounds)
		.def_readwrite("curve_samples", &ImplicitPart::curve_samples)
		.def_readwrite("valid", &ImplicitPart::valid)
		.def_readwrite("scale", &ImplicitPart::scale)
		.def_readwrite("translation", &ImplicitPart::translation);

	// hybrid_part.h
	py::class_<HybridPart>(m, "HybridPart")
		.def(py::init<const std::string&, const int, const int, const bool>())
		.def("ApplyTransform", &HybridPart::ApplyTransform)
		.def_readwrite("bounding_box", &HybridPart::bounding_box)
		.def_readwrite("face_surfaces", &HybridPart::face_surfaces)
		.def_readwrite("face_surface_parameters", &HybridPart::face_surface_parameters)
		.def_readwrite("face_surface_flipped", &HybridPart::face_surface_flipped)
		.def_readwrite("loop_types", &HybridPart::loop_types)
		.def_readwrite("loop_length", &HybridPart::loop_length)
		.def_readwrite("edge_curves", &HybridPart::edge_curves)
		.def_readwrite("edge_curve_parameters", &HybridPart::edge_curve_parameters)
		.def_readwrite("edge_curve_flipped", &HybridPart::edge_curve_flipped)
		.def_readwrite("edge_length", &HybridPart::edge_length)
		.def_readwrite("vertex_positions", &HybridPart::vertex_positions)
		.def_readwrite("face_to_face", &HybridPart::face_to_face)
		.def_readwrite("face_to_loop", &HybridPart::face_to_loop)
		.def_readwrite("loop_to_edge", &HybridPart::loop_to_edge)
		.def_readwrite("face_to_edge", &HybridPart::face_to_edge)
		.def_readwrite("edge_to_vertex", &HybridPart::edge_to_vertex)
		.def_readwrite("loop_to_vertex", &HybridPart::loop_to_vertex)
		.def_readwrite("edge_to_vertex_is_start", &HybridPart::edge_to_vertex_is_start)
		.def_readwrite("loop_to_edge_flipped", &HybridPart::loop_to_edge_flipped)
		.def_readwrite("face_to_edge_flipped", &HybridPart::face_to_edge_flipped)
		.def_readwrite("surface_bounds", &HybridPart::surface_bounds)
		.def_readwrite("surface_coords", &HybridPart::surface_coords)
		.def_readwrite("surface_samples", &HybridPart::surface_samples)
		.def_readwrite("curve_bounds", &HybridPart::curve_bounds)
		.def_readwrite("curve_samples", &HybridPart::curve_samples)
		.def_readwrite("valid", &HybridPart::valid)
		.def_readwrite("scale", &HybridPart::scale)
		.def_readwrite("translation", &HybridPart::translation);
}