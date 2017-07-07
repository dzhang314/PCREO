!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo_sphere - Point Configuration Riesz Energy Optimizer                    !
!                Optimized for Use on Unit Hyperspheres                        !
!                                                                              !
! PCreo_Sphere is a Fortran 2008 program that generates point configurations   !
! on the unit d-sphere embedded in (d+1)-dimensional Euclidean space and       !
! optimizes their Riesz s-energy, using either gradient descent or the BFGS    !
! algorithm. It differs from the generic version of PCreo in that certain      !
! optimizations have been performed which only make sense for unit             !
! (hyper)spheres.                                                              !
!                                                                              !
! In particular, because constraint equation of a sphere is so simple          !
! (x^2 + y^2 + z^2 = R^2), the surface_parameters module has been eliminated,  !
! and the calculations of constraint values and gradients have been inlined    !
! wherever they are needed. This saves numerous unnecessary subroutine calls.  !
!                                                                              !
! PCreo_Sphere offers numerous compile-time options which are configurable via !
! Fortran preprocessor (fpp) directives. The following preprocessor            !
! identifiers may be defined to configure PCreo_Sphere:                        !
!                                                                              !
!   PCREO_GRAD_DESC   -- Use gradient descent to optimize point                !
!                        configuations. If not defined, PCreo_Sphere defaults  !
!                        to the BFGS algorithm.                                !
!   PCREO_USE_MKL     -- Use the Intel Math Kernel Libraries to speed up       !
!                        matrix arithmetic. Requires use of the Intel Fortran  !
!                        compiler.                                             !
!   PCREO_SINGLE_PREC -- Use single-precision (32-bit) floating-point numbers  !
!                        to represent and optimize point configurations.       !
!                        If not defined, PCreo_Sphere defaults to              !
!                        double-precision (64-bit) floating point numbers.     !
!   PCREO_QUAD_PREC   -- Use quad-precision (128-bit) floating-point numbers   !
!                        to represent and optimize point configurations.       !
!                        If not defined, PCreo_Sphere defaults to              !
!                        double-precision (64-bit) floating point numbers.     !
!                        This option takes precedence over PCREO_SINGLE_PREC   !
!                        if both are simultaneously defined.                   !
!   PCREO_TRACK_ANGLE -- Track the normalized dot product between the step     !
!                        directions of the previous two iterations of gradient !
!                        descent or BFGS. This measures how much they overlap, !
!                        giving an indication of how sharply the step          !
!                        direction is turning. +1 indicates no turning at all, !
!                        while -1 indicates a full 180-degree reversal.        !
!                        This value is displayed as an extra column on the     !
!                        optimization status table printed to the console.     !
!   PCREO_SYMMETRY                                                             !
!                                                                              !
! Note that it is only necessary to #define these identifiers to enable their  !
! associated options. Their values do not matter, so even if defined as 0 or   !
! .false., they will still be active.                                          !
!                                                                              !
! For performance reasons, it is highly recommended to use a recent version of !
! the Intel Fortran compiler, with full optimizations enabled, to compile      !
! PCreo_Sphere. At the time of writing (Summer 2017), Intel Fortran is able    !
! to make far more extensive vectorization optimizations than its competitors, !
! beating (say) GFortran by roughly an order of magnitude.                     !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#if defined(PCREO_USE_MKL) .and. defined(PCREO_QUAD_PREC)
#error "Intel MKL does not support quad-precision arithmetic."
#endif

#if defined(PCREO_GRAD_DESC)
#undef PCREO_BFGS
#else
#define PCREO_BFGS
#endif

#if defined(PCREO_SINGLE_PREC) .or. defined(PCREO_QUAD_PREC)
#undef PCREO_DOUBLE_PREC
#else
#define PCREO_DOUBLE_PREC
#endif



module constants !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The constants module contains basic constants and parameters used throughout !
! PCreo_Sphere. All code to follow depends on this module.                     !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use, intrinsic :: iso_fortran_env, only : real32, real64, real128
    implicit none

#if defined(PCREO_SINGLE_PREC)
    integer, parameter :: rk = real32
    character(len=8), parameter :: rf = 'ES15.8E2'
#elif defined(PCREO_DOUBLE_PREC)
    integer, parameter :: rk = real64
    character(len=9), parameter :: rf = 'ES24.16E3'
#elif defined(PCREO_QUAD_PREC)
    integer, parameter :: rk = real128
    character(len=9), parameter :: rf = 'ES44.35E4'
#endif

    real(rk), parameter :: s = 1.0_rk
    integer, parameter :: d = 2
    integer, parameter :: num_points = 27
    real(rk), parameter :: print_time = 0.1_rk ! print 10 times per second
    real(rk), parameter :: save_time = 15.0_rk ! save every 15 seconds

#ifdef PCREO_USE_MKL
    include "mkl_blas.fi"
    integer, parameter :: num_vars = (d + 1) * num_points
#endif

#ifdef PCREO_SYMMETRY
    include "../include/icosahedral_symmetry_group.f90"
    include "../include/icosahedron_vertices.f90"
    integer, parameter :: num_external_points = size(external_points, 2)
    integer, parameter :: symmetry_group_order = size(symmetry_group, 3)
#endif

contains

    subroutine print_parameters
        character(len=80) :: line
        write(line,*) d
        write(*,*) "Sphere dimension: ", trim(adjustl(line))
        write(line,*) d + 1
        write(*,*) "Embedded in dimension: ", trim(adjustl(line))
        write(line,*) s
        write(*,*) "Value of s (Riesz potential power parameter): ", &
                & trim(adjustl(line))
#ifdef PCREO_SYMMETRY
        write(line,*) num_points
        write(*,*) "Number of movable points: ", trim(adjustl(line))
        write(line,*) num_external_points
        write(*,*) "Number of fixed points: ", trim(adjustl(line))
        write(line,*) external_energy
        write(*,*) "Fixed point energy: ", trim(adjustl(line))
        write(line,*) symmetry_group_order
        write(*,*) "Order of symmetry group: ", trim(adjustl(line))
        write(line,*) num_external_points + symmetry_group_order * num_points
        write(*,*) "Total number of points: ", trim(adjustl(line))
#else
        write(line,*) num_points
        write(*,*) "Number of points: ", trim(adjustl(line))
#endif
        write(line,*) print_time
        write(*,*) "Terminal output frequency: every ", &
                & trim(adjustl(line)), " seconds"
        write(line,*) save_time
        write(*,*) "File output frequency: every ", &
                & trim(adjustl(line)), " seconds"
    end subroutine print_parameters

end module constants



module system_utilities  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The system_utilities module contains subroutines for interacting with        !
! components of the Fortran runtime system (e.g. seeding the pseudorandom      !
! number generator).                                                           !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

contains

    subroutine init_random_seed
        integer :: i, n, clock
        integer, allocatable :: seed(:)

        call random_seed(size=n)
        allocate(seed(n))
        call system_clock(count=clock)
        seed = clock + 37 * (/ (i, i = 1, n) /)
        call random_seed(put=seed)
        deallocate(seed)
    end subroutine init_random_seed


    subroutine random_normal_points(points)
        real(rk), intent(out) :: points(:,:)

        real(rk), allocatable :: u(:,:)

        allocate(u, mold=points)
        call random_number(u)
        points = sqrt(-2.0_rk * log(u))
        call random_number(u)
        points = points * sin(4.0_rk * asin(1.0_rk) * u)
    end subroutine random_normal_points


    real(rk) function current_time()
        integer :: ticks, tick_rate
        call system_clock(ticks, tick_rate)
        current_time = real(ticks, rk) / real(tick_rate, rk)
    end function current_time

end module system_utilities



module pcreo_utilities  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The pcreo_utilities module contains subroutines for printing the current     !
! optimization status to the terminal and saving point configurations as       !
! comma-separated (csv) text files.                                            !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

contains

    subroutine print_welcome_message
        write(*,*) " _____   _____"
        write(*,*) "|  __ \ / ____|                David Zhang"
#if defined(PCREO_SYMMETRY)
        write(*,*) "| |__) | |     _ __ ___  ___    Symmetric"
#else
        write(*,*) "| |__) | |     _ __ ___  ___"
#endif
#if defined(PCREO_GRAD_DESC)
        write(*,*) "|  ___/| |    | '__/ _ \/ _ \  Grad. Desc."
#elif defined(PCREO_BFGS)
        write(*,*) "|  ___/| |    | '__/ _ \/ _ \    B F G S"
#endif
        write(*,*) "| |    | |____| | |  __/ (_) |  Optimized"
        write(*,*) "|_|     \_____|_|  \___|\___/  For Spheres"
    end subroutine print_welcome_message


    subroutine print_table_header
        if (rk == real32) then
            write(*,'(A)') "#Iterations| Riesz s-energy |&
                    &  RMS Gradient  | Step size"
            write(*,'(A)') "-----------+----------------+&
                    &----------------+----------------"
        else if (rk == real64) then
            write(*,'(A)') "#Iterations| Riesz s-energy         |&
                    & RMS Gradient           | Step size"
            write(*,'(A)') "-----------+-------------------------+&
                    &------------------------+------------------------"
        else if (rk == real128) then
            write(*,'(A)') "#Iterations|&
                    & Riesz s-energy                              |&
                    & RMS Gradient                                |&
                    & Step size"
            write(*,'(A)') "-----------+&
                    &---------------------------------------------+&
                    &---------------------------------------------+&
                    &---------------------------------------------"
        end if
    end subroutine print_table_header


    subroutine save_point_file(points, idx)
        real(rk), intent(in) :: points(d + 1, num_points)
        integer, intent(in) :: idx

        integer :: i, j, u
        character(len=34) :: fname

        write(fname,"(a,i10.10,a)") "point_configuration_", idx, ".csv"
        open(file=fname, newunit=u)
        do j = 1, num_points
            do i = 1, d + 1
                if (i > 1) write(u,'(A)',advance="no") ", "
                write(u,'(SP,'//rf//')',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end module pcreo_utilities



module sphere_riesz_energy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The sphere_riesz_energy module contains subroutines for generating random    !
! point configurations on the unit (hyper)sphere and computing their Riesz     !
! s-energy.                                                                    !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

contains

    pure subroutine constrain_points(points)
        real(rk), intent(inout) :: points(d + 1, num_points)

        integer :: i

        do i = 1, num_points
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points

#ifdef PCREO_SYMMETRY

    pure real(rk) function pair_potential(r)
        real(rk), intent(in) :: r

        pair_potential = r**(-s)
    end function pair_potential


    pure real(rk) function pair_potential_derivative(r)
        real(rk), intent(in) :: r

        pair_potential_derivative = -s * r**(-s - 1.0_rk)
    end function pair_potential_derivative


    pure subroutine add_pair_energy(source_pt, target_pt, energy)
        real(rk), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(rk), intent(inout) :: energy

        energy = energy + pair_potential(norm2(target_pt - source_pt))
    end subroutine add_pair_energy


    pure subroutine add_pair_energy_2(source_pt, target_pt, energy)
        real(rk), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(rk), intent(inout) :: energy

        energy = energy + 2.0_rk * pair_potential(norm2(target_pt - source_pt))
    end subroutine add_pair_energy_2


    pure subroutine add_pair_energy_force(source_pt, target_pt, energy, force)
        real(rk), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(rk), intent(inout) :: energy, force(d + 1)

        real(rk) :: displacement(d + 1), r

        displacement = target_pt - source_pt
        r = norm2(displacement)
        energy = energy + pair_potential(r)
        force = force - (pair_potential_derivative(r) / r) * displacement
    end subroutine add_pair_energy_force


    pure subroutine add_pair_energy_force_2(source_pt, target_pt, energy, force)
        real(rk), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(rk), intent(inout) :: energy, force(d + 1)

        real(rk) :: displacement(d + 1), r

        displacement = target_pt - source_pt
        r = norm2(displacement)
        energy = energy + 2.0_rk * pair_potential(r)
        force = force - (pair_potential_derivative(r) / r) * displacement
    end subroutine add_pair_energy_force_2


    pure real(rk) function riesz_energy(points)
        real(rk), intent(in) :: points(d + 1, num_points)

        integer :: b, p, q

        riesz_energy = 0.0_rk
        do p = 1, num_points
            do q = 1, num_external_points
                call add_pair_energy_2( &
                    & external_points(:,q), &
                    & points(:,p), riesz_energy)
            end do
            do b = 2, symmetry_group_order
                call add_pair_energy( &
                    & matmul(symmetry_group(:,:,b), points(:,p)), &
                    & points(:,p), riesz_energy)
            end do
            do q = 1, p - 1
                do b = 1, symmetry_group_order
                    call add_pair_energy( &
                        & matmul(symmetry_group(:,:,b), points(:,q)), &
                        & points(:,p), riesz_energy)
                end do
            end do
            do q = p + 1, num_points
                do b = 1, symmetry_group_order
                    call add_pair_energy( &
                        & matmul(symmetry_group(:,:,b), points(:,q)), &
                        & points(:,p), riesz_energy)
                end do
            end do
        end do
        riesz_energy = external_energy + &
            & 0.5_rk * symmetry_group_order * riesz_energy
    end function riesz_energy


    pure subroutine riesz_energy_force(points, energy, force)
        real(rk), intent(in) :: points(d + 1, num_points)
        real(rk), intent(out) :: energy, force(d + 1, num_points)

        integer :: b, p, q

        energy = 0.0_rk
        do p = 1, num_points
            force(:,p) = 0.0_rk
            do q = 1, num_external_points
                call add_pair_energy_force_2( &
                    & external_points(:,q), &
                    & points(:,p), energy, force(:,p))
            end do
            do b = 2, symmetry_group_order
                call add_pair_energy_force( &
                    & matmul(symmetry_group(:,:,b), points(:,p)), &
                    & points(:,p), energy, force(:,p))
            end do
            do q = 1, p - 1
                do b = 1, symmetry_group_order
                    call add_pair_energy_force( &
                        & matmul(symmetry_group(:,:,b), points(:,q)), &
                        & points(:,p), energy, force(:,p))
                end do
            end do
            do q = p + 1, num_points
                do b = 1, symmetry_group_order
                    call add_pair_energy_force( &
                        & matmul(symmetry_group(:,:,b), points(:,q)), &
                        & points(:,p), energy, force(:,p))
                end do
            end do
            force(:,p) = force(:,p) - &
                & dot_product(force(:,p), points(:,p)) * points(:,p)
        end do
        force = symmetry_group_order * force
        energy = external_energy + 0.5_rk * symmetry_group_order * energy
    end subroutine riesz_energy_force

#else

    pure real(rk) function riesz_energy(points)
        real(rk), intent(in) :: points(:,:)

        integer :: i, j

        riesz_energy = 0.0_rk
        do j = 1, size(points, 2)
            do i = 1, j - 1
                riesz_energy = riesz_energy + &
                        & norm2(points(:,i) - points(:,j))**(-s)
            end do
        end do
    end function riesz_energy


    pure subroutine riesz_energy_force(points, ener, force)
        real(rk), intent(in) :: points(:,:)
        real(rk), intent(out) :: ener, force(:,:)

        real(rk) :: displ(size(points, 1)), dist_sq, term
        integer :: i, j

        ener = 0.0_rk
        do j = 1, size(points, 2)
            force(:,j) = 0.0_rk
            do i = 1, j - 1
                displ = points(:,i) - points(:,j)
                dist_sq = dot_product(displ, displ)
                term = dist_sq**(-0.5_rk * s)
                ener = ener + term
                term = s * term / dist_sq
                force(:,j) = force(:,j) - term * displ
            end do
            do i = j + 1, size(points, 2)
                displ = points(:,i) - points(:,j)
                term = s * norm2(displ)**(-s - 2.0_rk)
                force(:,j) = force(:,j) - term * displ
            end do
            force(:,j) = force(:,j) - &
                & dot_product(force(:,j), points(:,j)) * points(:,j)
        end do
    end subroutine riesz_energy_force

#endif

end module sphere_riesz_energy



module linear_algebra_4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
!                                                                              !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

contains

    real(rk) function dot_product_2(v, w) result (dot)
        real(rk), intent(in), contiguous, dimension(d + 1, num_points) :: v, w

#if defined(PCREO_USE_MKL) .and. defined(PCREO_SINGLE_PREC)
        dot = sdot(num_vars, v, 1, w, 1)
#elif defined(PCREO_USE_MKL) .and. defined(PCREO_DOUBLE_PREC)
        dot = ddot(num_vars, v, 1, w, 1)
#else
        integer :: i, j

        dot = 0.0_rk
        do j = 1, num_points
            do i = 1, d + 1
                dot = dot + v(i,j) * w(i,j)
            end do
        end do
#endif
    end function dot_product_2


    subroutine matrix_multiply_42(c, A, b)
        real(rk), intent(out), contiguous :: c(d + 1, num_points)
        real(rk), intent(in), contiguous :: &
                & A(d + 1, num_points, d + 1, num_points), b(d + 1, num_points)

#if defined(PCREO_USE_MKL) .and. defined(PCREO_SINGLE_PREC)
        call ssymv('U', num_vars, 1.0_rk, A, num_vars, b, 1, 0.0_rk, c, 1)
#elif defined(PCREO_USE_MKL) .and. defined(PCREO_DOUBLE_PREC)
        call dsymv('U', num_vars, 1.0_rk, A, num_vars, b, 1, 0.0_rk, c, 1)
#else
        integer :: i, j, k, l

        do j = 1, num_points
            do i = 1, d + 1
                c(i,j) = 0.0_rk
                do l = 1, num_points
                    do k = 1, d + 1
                        c(i,j) = c(i,j) + A(k,l,i,j) * b(k,l)
                    end do
                end do
            end do
        end do
#endif
    end subroutine matrix_multiply_42


    subroutine symmetric_update_4(A, c, x, y)
        real(rk), intent(inout), contiguous :: &
                & A(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in) :: c
        real(rk), intent(in), contiguous :: &
                & x(d + 1, num_points), y(d + 1, num_points)

#if defined(PCREO_USE_MKL) .and. defined(PCREO_SINGLE_PREC)
        call ssyr2('U', num_vars, c, x, 1, y, 1, A, num_vars)
#elif defined(PCREO_USE_MKL) .and. defined(PCREO_DOUBLE_PREC)
        call dsyr2('U', num_vars, c, x, 1, y, 1, A, num_vars)
#else
        integer :: i, j, k, l

        do concurrent (i = 1 : d + 1, j = 1 : num_points, &
                     & k = 1 : d + 1, l = 1 : num_points)
            A(i,j,k,l) = A(i,j,k,l) + c * (x(i,j) * y(k,l) + y(i,j) * x(k,l))
        end do
#endif
    end subroutine symmetric_update_4


    subroutine identity_matrix_4(A)
        real(rk), intent(out), contiguous :: &
                & A(d + 1, num_points, d + 1, num_points)

#if defined(PCREO_USE_MKL) .and. defined(PCREO_SINGLE_PREC)
        call slaset('U', num_vars, num_vars, 0.0_rk, 1.0_rk, A, num_vars)
#elif defined(PCREO_USE_MKL) .and. defined(PCREO_DOUBLE_PREC)
        call dlaset('U', num_vars, num_vars, 0.0_rk, 1.0_rk, A, num_vars)
#else
        integer :: i, j, k, l

        do concurrent (i = 1 : d + 1, j = 1 : num_points, &
                     & k = 1 : d + 1, l = 1 : num_points)
            A(i,j,k,l) = merge(1.0_rk, 0.0_rk, i == k .and. j == l)
        end do
#endif
    end subroutine identity_matrix_4

end module linear_algebra_4



module optimization_subroutines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
!                                                                              !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use linear_algebra_4
    implicit none

contains

    subroutine bfgs_update_inverse_hessian( &
            & inv_hess, delta_points, delta_gradient)
        real(rk), intent(inout) :: &
                & inv_hess(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in), dimension(d + 1, num_points) :: &
                & delta_points, delta_gradient

        real(rk) :: lambda, theta, sigma, kappa(d + 1, num_points)

        lambda = dot_product_2(delta_gradient, delta_points)
        call matrix_multiply_42(kappa, inv_hess, delta_gradient)
        theta = dot_product_2(delta_gradient, kappa)
        sigma = (lambda + theta) / (lambda * lambda)
        kappa = kappa - (0.5_rk * lambda * sigma) * delta_points
        call symmetric_update_4(inv_hess, -1 / lambda, kappa, delta_points)
    end subroutine bfgs_update_inverse_hessian


    pure real(rk) function quadratic_line_search(points, energy, &
            & step_direction, initial_step_size) result (optimal_step_size)
        real(rk), intent(in) :: points(d + 1, num_points)
        real(rk), intent(in) :: energy
        real(rk), intent(in) :: step_direction(d + 1, num_points)
        real(rk), intent(in) :: initial_step_size

        real(rk), dimension(d + 1, num_points) :: new_points, newer_points
        real(rk) :: step_size, new_energy, newer_energy
        integer :: num_increases

        ! The goal of quadratic line search is to find three points, a, b, c,
        ! such that a < b < c and f(a) > f(b) < f(c). We say that such a
        ! triplet of points is "bowl-shaped." Once we have three bowl-shaped
        ! points, we fit a parabola through them, and take its minimum as our
        ! best step size. In this implementation, we take a = 0, so that f(a)
        ! is the energy at the initial point, and work to find b and c such
        ! that c = 2*b.

        ! First, we take a step using our initial step size, and see where it
        ! leads us. In particular, does it increase or decrease the energy?
        step_size = initial_step_size
        new_points = points + step_size * step_direction
        call constrain_points(new_points)
        new_energy = riesz_energy(new_points)

        ! If the new energy is less than the old energy, then we can afford
        ! to be a bit more ambitious. We try a larger step size.
        if (new_energy < energy) then
            num_increases = 0
            do
                ! Try taking a step of double the size. Does this result in
                ! an increase?
                newer_points = points + (2.0_rk * step_size) * step_direction
                call constrain_points(newer_points)
                newer_energy = riesz_energy(newer_points)
                ! If so, then we have our bowl-shaped points, and we exit
                ! the loop.
                if (newer_energy >= new_energy) then
                    exit
                ! If not, then we can be even more ambitious. Double the
                ! step size again.
                else
                    step_size = 2.0_rk * step_size
                    new_points = newer_points
                    new_energy = newer_energy
                    num_increases = num_increases + 1
                    ! We might run into a situation where increasing the step
                    ! size "accidentally" decreases the energy by, say, jumping
                    ! into the basin of a deeper local minimum. To prevent this
                    ! from getting us too far off track, we limit the number of
                    ! consecutive times the step size can increase.
                    if (num_increases >= 4) then
                        optimal_step_size = step_size
                        return
                    end if
                end if
            end do
            ! Finally, once we have our bowl-shaped points, we take the arg
            ! min of the interpolating parabola. The formula, worked out in
            ! advance, is as follows:
            optimal_step_size = 0.5_rk * step_size * &
                    & (4.0_rk * new_energy - newer_energy - 3.0_rk * energy) / &
                    & (2.0_rk * new_energy - newer_energy - energy)
            ! Note that this formula is numerically unstable, since it contains
            ! subtractions of roughly equal-magnitude numbers that can result
            ! in catastrophic cancellation. To check whether this has occurred,
            ! we perform one last sanity check: the arg min should fall
            ! somewhere inside the bowl.
            if (0.0_rk < optimal_step_size .and. &
                    & optimal_step_size < 2.0_rk * step_size) then
                return
            ! If our sanity check has failed, then the bowl we found must be so
            ! shallow that it doesn't really matter what step size we return.
            ! Just take the middle of the bowl.
            else
                optimal_step_size = step_size
                return
            end if
        ! Now, if the new energy is greater than the old energy, or worse,
        ! gives a non-finite (Inf/NaN) result, then we know our initial step
        ! size was too large.
        else
            do
                ! Try taking a step of half the size. Does this result in a
                ! decrease?
                newer_points = points + (0.5_rk * step_size) * step_direction
                call constrain_points(newer_points)
                newer_energy = riesz_energy(newer_points)
                ! If so, then we have our bowl-shaped points, and we exit
                ! the loop.
                if (newer_energy < energy) then
                    exit
                ! Otherwise, we need to halve the step size and try again.
                else
                    step_size = 0.5_rk * step_size
                    ! If no step produces a decrease, no matter how small, then
                    ! we have probably started our search from a local minimum.
                    ! Return zero step size to indicate this.
                    if (abs(step_size) < epsilon(1.0_rk)) then
                        optimal_step_size = 0.0_rk
                        return
                    end if
                    new_points = newer_points
                    new_energy = newer_energy
                end if
            end do
            ! Again, we use the following formula for the arg min of the
            ! interpolating parabola. Note that this is slightly different than
            ! the previous one -- here we have b = step_size/2, c = step_size,
            ! whereas before we had b = step_size, c = 2*step_size.
            optimal_step_size = 0.25_rk * step_size * &
                    & (new_energy - 4.0_rk * newer_energy + 3.0_rk * energy) / &
                    & (new_energy - 2.0_rk * newer_energy + energy)
            ! We perform a similar sanity check to guard against numerical
            ! instability.
            if (0.0_rk < optimal_step_size .and. &
                    & optimal_step_size < step_size) then
                return
            else
                optimal_step_size = 0.5_rk * step_size
                return
            end if
        end if
    end function quadratic_line_search

end module optimization_subroutines



program pcreo_sphere !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Main program of PCreo_Sphere. Contains subroutines for displaying and saving !
! the current optimization status.                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use linear_algebra_4
    use optimization_subroutines
    use system_utilities
    use pcreo_utilities
    implicit none

    real(rk) :: points(d + 1, num_points)
    real(rk) :: energy, force(d + 1, num_points)
    real(rk) :: step_size
    real(rk) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

#if defined(PCREO_GRAD_DESC) .and. defined(PCREO_TRACK_ANGLE)
    real(rk) :: step_angle
    real(rk) :: new_force(d + 1, num_points)
#elif defined(PCREO_BFGS)
    real(rk) :: new_points(d + 1, num_points)
    real(rk) :: new_energy, new_force(d + 1, num_points)
    real(rk) :: step_direction(d + 1, num_points)
    real(rk) :: inv_hess(d + 1, num_points, d + 1, num_points)
    real(rk), dimension(d + 1, num_points) :: delta_points, delta_gradient
#ifdef PCREO_TRACK_ANGLE
    real(rk) :: step_angle
    real(rk) :: new_step_direction(d + 1, num_points)
#endif
#endif

    call print_welcome_message
    write(*,*)
    call print_parameters
    write(*,*)
    call initialize_point_configuration(points, energy, force)
    write(*,*)

#ifdef PCREO_BFGS
    call identity_matrix_4(inv_hess)
    step_direction = force
#endif
    step_size = 1.0E-10_rk
#ifdef PCREO_TRACK_ANGLE
    step_angle = 0.0_rk
#endif

    iteration_count = 0
    cur_time = current_time()
    last_print_time = cur_time
    last_save_time = cur_time

    call print_table_header
    call print_optimization_status

#if defined(PCREO_GRAD_DESC)

    do
        step_size = quadratic_line_search(points, energy, force, step_size)
        call check_step_size
        points = points + step_size * force
        call constrain_points(points)
#ifdef PCREO_TRACK_ANGLE
        call riesz_energy_force(points, energy, new_force)
        step_angle = dot_product_2(force, new_force) / &
                & (norm2(force) * norm2(new_force))
        force = new_force
#else
        call riesz_energy_force(points, energy, force)
#endif
        call finish_iteration
    end do

#elif defined(PCREO_BFGS)

    do
        step_size = quadratic_line_search( &
                & points, energy, step_direction, step_size)
        call check_step_size
        new_points = points + step_size * step_direction
        call constrain_points(new_points)
        delta_points = new_points - points
        call riesz_energy_force(new_points, new_energy, new_force)
        delta_gradient = force - new_force
        call bfgs_update_inverse_hessian(inv_hess, delta_points, delta_gradient)
        points = new_points
        energy = new_energy
        force = new_force
#ifdef PCREO_TRACK_ANGLE
        call matrix_multiply_42(new_step_direction, inv_hess, force)
        if (iteration_count > 0) then
            step_angle = dot_product_2(step_direction, new_step_direction) / &
                    & (norm2(step_direction) * norm2(new_step_direction))
        end if
        step_direction = new_step_direction
#else
        call matrix_multiply_42(step_direction, inv_hess, force)
#endif
        call finish_iteration
    end do

#endif


contains

    subroutine initialize_point_configuration(points, energy, force)
        real(rk), intent(out) :: points(d + 1, num_points)
        real(rk), intent(out) :: energy, force(d + 1, num_points)
        integer :: u
        logical :: ex

        inquire(file="initial_configuration.txt", exist=ex)
        if (ex) then
            write(*,*) "Loading initial point configuration from file..."
            open(newunit=u, file="initial_configuration.txt")
            read(u,*) points
            close(u)
        else
            write(*,*) "Generating random initial point configuration..."
            call init_random_seed
            call random_normal_points(points)
        end if
        call constrain_points(points)
        call riesz_energy_force(points, energy, force)
        write(*,*) "Point configuration initialized."
    end subroutine initialize_point_configuration


    subroutine print_optimization_status
        write(*,'(I10,A)',advance="no") iteration_count, " |"
        write(*,'('//rf//',A)',advance="no") energy, " |"
        write(*,'('//rf//',A)',advance="no") &
                & norm2(force) / sqrt(real(num_points, rk)), " |"
#ifdef PCREO_TRACK_ANGLE
        write(*,'('//rf//',A)',advance="no") step_size, " |"
        write(*,'('//rf//')') step_angle
#else
        write(*,'('//rf//')') step_size
#endif
    end subroutine print_optimization_status


    subroutine check_step_size
        if (step_size == 0.0_rk) then
            call print_optimization_status
            call save_point_file(points, iteration_count)
            write(*,*) "Convergence has been achieved (up to numerical&
                    & round-off error). Exiting."
            stop
        end if
    end subroutine check_step_size


    subroutine finish_iteration
        iteration_count = iteration_count + 1
        cur_time = current_time()
        if (cur_time - last_print_time >= print_time) then
            call print_optimization_status
            last_print_time = cur_time
        end if
        if (cur_time - last_save_time >= save_time) then
            call save_point_file(points, iteration_count)
            last_save_time = cur_time
        end if
    end subroutine finish_iteration

end program pcreo_sphere
