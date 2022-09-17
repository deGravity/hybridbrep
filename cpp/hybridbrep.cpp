#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "implicit_part.h"

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
		.def_readwrite("valid", &ImplicitPart::valid);
}