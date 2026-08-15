// hellofem::nls — explicit instantiations
// SPDX-License-Identifier: MIT

#include "NewtonSolver.h"

namespace hellofem::nls {

    template class NewtonSolver<double>;
    template class NewtonSolver<float>;

} // namespace hellofem::nls
