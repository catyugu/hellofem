// hellofem::io — VTK/ParaView file output
// SPDX-License-Identifier: MIT

#pragma once

#include "io/vtk_utils.h"
#include "mesh/Mesh.h"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hellofem::fem {
    template <std::floating_point T>
    class Function;
}

namespace hellofem::io {

    /// Output of meshes and functions in VTK/ParaView format.
    ///
    /// Each `write` call stores one ascii `.vtu` file; a `.pvd` index of
    /// the time steps is written on `close`. Intended for visualization
    /// only (not for checkpointing).
    class VTKFile {
    public:
        /// Create a VTK file series.
        ///
        /// @param[in] filename Base path (e.g. `"results.pvd"`); the
        /// per-step `.vtu` files are derived from it.
        /// @param[in] file_mode Output mode: `"w"` truncates, `"a"`
        /// appends to an existing series.
        VTKFile(const std::filesystem::path& filename,
            std::string_view file_mode = "w");

        /// Write the `.pvd` index and close the series.
        ~VTKFile();

        /// Write the `.pvd` index (called automatically by the
        /// destructor).
        void close();

        /// Flush buffered state. No-op (files are written directly).
        void flush() { }

        /// Write a mesh at time `t`.
        void write(const mesh::Mesh<double>& mesh, double t = 0.0);

        /// Write functions at time `t`.
        ///
        /// @pre All functions must be scalar (block size 1) Lagrange
        /// functions defined on the same function space (the first
        /// function's space provides the mesh and node ordering).
        void write(const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
            double t);

    private:
        /// Base output path (e.g. `"results.pvd"`).
        std::filesystem::path _filename;
        /// Directory of the base path.
        std::filesystem::path _directory;
        /// Base stem of the per-step files.
        std::string _stem;
        /// Accumulated (time, vtu file name) steps.
        std::vector<std::pair<double, std::string>> _steps;
        /// Number of steps written.
        std::size_t _counter = 0;
    };

} // namespace hellofem::io
