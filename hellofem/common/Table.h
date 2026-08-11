// hellofem::common — Table: name-keyed pretty-printed value table
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace hellofem {

    /// Storage and pretty-printing for tables of values keyed by (row, col).
    ///
    ///   Table table("Timings");
    ///   table.set("Foo", "Assemble", 0.010);
    ///   table.set("Foo", "Solve", 0.020);
    class Table {
    public:
        /// Reduction flavour (min/max/average) — retained for API parity with
        /// the timings interface, though a single-process library never
        /// aggregates across ranks.
        enum class Reduction : std::uint8_t { average, max, min };

        Table(std::string title = "", bool right_justify = true);

        /// Copy constructor
        Table(const Table& table) = default;

        /// Move constructor
        Table(Table&& table) = default;

        /// Destructor
        ~Table() = default;

        /// Copy assignment
        Table& operator=(const Table& table) = default;

        /// Move assignment
        Table& operator=(Table&& table) = default;

        /// Set table entry.
        void set(std::string_view row, std::string_view col,
                 std::variant<std::string, int, double> value);

        /// Get table entry; throws if the entry does not exist.
        std::variant<std::string, int, double>
        get(std::string_view row, std::string_view col) const;

        /// Table title.
        std::string name;

        /// String representation of the table.
        std::string str() const;

    private:
        // Row and column names, in insertion order
        std::vector<std::string> _rows, _cols;

        // Table entry values
        std::map<std::pair<std::string, std::string>,
                 std::variant<std::string, int, double>>
            _values;

        // Right-justify the table entries
        bool _right_justify;
    };

} // namespace hellofem
