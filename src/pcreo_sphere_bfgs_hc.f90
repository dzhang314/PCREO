#define PCREO_USE_MKL
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo_sphere_bfgs_hc - Point Configuration Riesz Energy Optimizer            !
!                        Optimized for Use on Unit Hyperspheres                !
!                        BFGS Version                                          !
!                        Hardcoded Inputs                                      !
!                                                                              !
! This program generates point configurations on the unit d-sphere embedded in !
! (d+1)-dimensional Euclidean space and optimizes their Riesz s-energy by the  !
! BFGS algorithm. It differs from the generic version of pcreo in that certain !
! structural optimizations have been performed which only make sense for unit  !
! (hyper)spheres.                                                              !
!                                                                              !
! In particular, because constraint equation of a sphere is so simple          !
! (x^2 + y^2 + z^2 = R^2), the surface_parameters module has been eliminated,  !
! and the calculations of constraint values and gradients have been inlined    !
! wherever they are needed. This saves numerous unnecessary subroutine calls.  !
!                                                                              !
! In addition, startup time is significantly improved by implementing a faster !
! random point selection algorithm based on the Box-Muller transform, which    !
! involves no trial-and-error. (The original algorithm needed to reject points !
! which did not correctly project down to the desired surface.)                !
!                                                                              !
! Note that, unlike the generic version of pcreo, this program uses MKL BLAS   !
! subroutines throughout to speed up the matrix arithmetic required for the    !
! BFGS algorithm. It is therefore highly recommended to compile this program   !
! with a recent version of the Intel Fortran compiler, with full optimizations !
! enabled, against a recent version of the Intel MKL libraries.                !
!                                                                              !
! In addition, because BLAS subroutines are not kind-generic (i.e. dgemv can   !
! only multiply double precision matrices, not single precision), this version !
! of pcreo is specialized to double-precision calculations, and does not offer !
! a single global precision switch.                                            !
!                                                                              !
! Note also that this program does not accept command-line arguments. Instead, !
! all parameters (number of points, value of s, etc.) have been made           !
! compile-time constants, hardcoded in the constants module below. This        !
! provides a significant speed boost with the Intel Fortran compiler, at the   !
! inconvenience of having to re-compile for every set of parameters.           !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



module constants !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The constants module contains basic constants and parameters used throughout !
! the rest of the program.                                                     !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    implicit none

#ifdef PCREO_USE_MKL
#include "mkl_blas.fi"
    integer, parameter :: rk = selected_real_kind(15, 307)  ! IEEE double prec.
#else
    integer, parameter :: sp = selected_real_kind(6, 37)    ! IEEE single prec.
    integer, parameter :: dp = selected_real_kind(15, 307)  ! IEEE double prec.
    integer, parameter :: qp = selected_real_kind(33, 4931) ! IEEE quad prec.
    integer, parameter :: rk = dp
#endif

    real(rk), parameter :: s = 2.0_rk
    integer, parameter :: d = 3
    integer, parameter :: num_points = 100

    real(rk), parameter :: print_time = 0.1_rk ! print 10 times per second
    real(rk), parameter :: save_time = 15.0_rk ! save every 15 seconds

    integer, parameter :: num_vars = (d + 1) * num_points

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
        write(line,*) num_points
        write(*,*) "Number of points: ", trim(adjustl(line))
        write(line,*) print_time
        write(*,*) "Terminal output frequency: every ", &
                & trim(adjustl(line)), " seconds"
        write(line,*) save_time
        write(*,*) "File output frequency: every ", &
                & trim(adjustl(line)), " seconds"
    end subroutine print_parameters

end module constants



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

    subroutine random_normal_points(points)
        real(rk), intent(out) :: points(d + 1, num_points)

        real(rk) :: u(d + 1, num_points)

        call random_number(u)
        points = sqrt(-2.0_rk * log(u))
        call random_number(u)
        points = points * sin(4.0_rk * asin(1.0_rk) * u)
    end subroutine random_normal_points


    pure subroutine constrain_points(points)
        real(rk), intent(inout) :: points(d + 1, num_points)

        integer :: i

        do i = 1, num_points
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points


    pure real(rk) function riesz_energy(points)
        real(rk), intent(in) :: points(d + 1, num_points)

        integer :: i, j

        riesz_energy = 0.0_rk
        do j = 1, num_points
            do i = 1, j - 1
                riesz_energy = riesz_energy + &
                        & norm2(points(:,i) - points(:,j))**(-s)
            end do
        end do
    end function riesz_energy


    pure subroutine riesz_energy_gradient(points, ener, grad)
        real(rk), intent(in) :: points(d + 1, num_points)
        real(rk), intent(out) :: ener, grad(d + 1, num_points)

        real(rk) :: displ(d + 1), dist_sq, term
        integer :: i, j

        ener = 0.0_rk
        do j = 1, num_points
            grad(:,j) = 0.0_rk
            do i = 1, j - 1
                displ = points(:,i) - points(:,j)
                dist_sq = dot_product(displ, displ)
                term = dist_sq**(-0.5_rk * s)
                ener = ener + term
                term = s * term / dist_sq
                grad(:,j) = grad(:,j) + term * displ
            end do
            do i = j + 1, num_points
                displ = points(:,i) - points(:,j)
                term = s * norm2(displ)**(-s - 2.0_rk)
                grad(:,j) = grad(:,j) + term * displ
            end do
            grad(:,j) = grad(:,j) - &
                & dot_product(grad(:,j), points(:,j)) * points(:,j)
        end do
    end subroutine riesz_energy_gradient

end module sphere_riesz_energy



module linear_algebra_4

    use constants
    implicit none

contains

    real(rk) function dot_product_2(v, w) result (dot)
        real(rk), intent(in), dimension(d + 1, num_points) :: v, w

#ifdef PCREO_USE_MKL
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
        real(rk), intent(out) :: c(d + 1, num_points)
        real(rk), intent(in) :: A(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in) :: b(d + 1, num_points)

#ifdef PCREO_USE_MKL
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


    subroutine matrix_multiply_neg_42(c, A, b)
        real(rk), intent(out) :: c(d + 1, num_points)
        real(rk), intent(in) :: A(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in) :: b(d + 1, num_points)

#ifdef PCREO_USE_MKL
        call dsymv('U', num_vars, -1.0_rk, A, num_vars, b, 1, 0.0_rk, c, 1)
#else
        integer :: i, j, k, l

        do j = 1, num_points
            do i = 1, d + 1
                c(i,j) = 0.0_rk
                do l = 1, num_points
                    do k = 1, d + 1
                        c(i,j) = c(i,j) - A(k,l,i,j) * b(k,l)
                    end do
                end do
            end do
        end do
#endif
    end subroutine matrix_multiply_neg_42


    subroutine symmetric_update_4(A, c, x, y)
        real(rk), intent(inout) :: A(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in) :: c, x(d + 1, num_points), y(d + 1, num_points)

#ifdef PCREO_USE_MKL
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
        real(rk), intent(out) :: A(d + 1, num_points, d + 1, num_points)

#ifdef PCREO_USE_MKL
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



module bfgs_subroutines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The bfgs_subroutines module contains subroutines for the two key operations  !
! in the BFGS algorithm: line search, which in our case is a simple quadratic  !
! line search, and performing symmetric rank-two updates of the approximate    !
! inverse Hessian.                                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use linear_algebra_4
    implicit none

contains

    subroutine update_inverse_hessian( &
            & inv_hess, delta_gradient, step_size, step_direction)
        real(rk), intent(inout) :: &
                & inv_hess(d + 1, num_points, d + 1, num_points)
        real(rk), intent(in) :: delta_gradient(d + 1, num_points)
        real(rk), intent(in) :: step_size, step_direction(d + 1, num_points)

        real(rk) :: lambda, theta, sigma, kappa(d + 1, num_points)

        lambda = step_size * dot_product_2(delta_gradient, step_direction)
        call matrix_multiply_42(kappa, inv_hess, delta_gradient)
        theta = dot_product_2(delta_gradient, kappa)
        sigma = (lambda + theta) / (lambda * lambda)
        kappa = kappa - (0.5_rk * step_size * lambda * sigma) * step_direction
        call symmetric_update_4(inv_hess, -step_size / lambda, &
                              & kappa, step_direction)
    end subroutine update_inverse_hessian

end module bfgs_subroutines



module line_search !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The line_search module contains a single subroutine implementing a simple    !
! quadratic line search algorithm.                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    implicit none

contains

    include "../include/quadratic_line_search.f90"

end module line_search



program pcreo_sphere_bfgs_hc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Main program of pcreo_sphere_bfgs_hc. Contains subroutines for displaying    !
! and saving the current optimization status.                                  !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use linear_algebra_4
    use line_search
    use bfgs_subroutines
    implicit none

    real(rk), dimension(d + 1, num_points) :: old_points, new_points
    real(rk) :: old_energy, new_energy
    real(rk), dimension(d + 1, num_points) :: old_gradient, new_gradient
    ! Approximate inverse Hessian, calculated by BFGS
    real(rk) :: inv_hess(d + 1, num_points, d + 1, num_points), step_size
    real(rk), dimension(d + 1, num_points) :: step_direction, delta_gradient
    real(rk) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

#ifdef PCREO_TRACK_ANGLE
    real(rk) :: step_angle
    real(rk), dimension(d + 1, num_points) :: new_step_direction
#endif

    call print_welcome_message
    write(*,*)
    call print_parameters
    write(*,*)
    call initialize_point_configuration(old_points, old_energy, old_gradient)
    write(*,*)

    ! Initialize inv_hess to identity matrix
    call identity_matrix_4(inv_hess)
    ! TODO: Is there a more natural choice of initial step size?
    step_size = 1.0E-10_rk

    iteration_count = 0
    cur_time = current_time()
    last_print_time = cur_time
    last_save_time = cur_time

    call print_table_header
    call print_optimization_status
    do
#ifdef PCREO_TRACK_ANGLE
        ! Multiply inverse hessian by negative gradient to obtain step direction
        call matrix_multiply_42(new_step_direction, inv_hess, old_gradient)
        if (iteration_count > 0) then
            step_angle = sum(step_direction * new_step_direction) / &
                    & (norm2(step_direction) * norm2(new_step_direction))
        else
            step_angle = 0.0_rk
        end if
        step_direction = new_step_direction
#else
        ! Multiply inverse hessian by negative gradient to obtain step direction
        call matrix_multiply_42(step_direction, inv_hess, old_gradient)
#endif
        step_size = quadratic_line_search(old_points, old_energy, &
                & step_direction, step_size)
        if (step_size == 0.0_rk) then
            call print_optimization_status
            call save_point_file(old_points, iteration_count)
            write(*,*) "Convergence has been achieved (up to numerical&
                    & round-off error). Exiting."
            stop
        end if
        new_points = old_points + step_size * step_direction
        call constrain_points(new_points)
        call riesz_energy_gradient(new_points, new_energy, new_gradient)
        delta_gradient = new_gradient - old_gradient
        call update_inverse_hessian(inv_hess, delta_gradient, &
                & step_size, step_direction)
        old_points = new_points
        old_energy = new_energy
        old_gradient = new_gradient
        iteration_count = iteration_count + 1
        cur_time = current_time()
        if (cur_time - last_print_time >= print_time) then
            call print_optimization_status
            last_print_time = cur_time
        end if
        if (cur_time - last_save_time >= save_time) then
            call save_point_file(old_points, iteration_count)
            last_save_time = cur_time
        end if
    end do

contains

    subroutine print_welcome_message
        write(*,*) " _____   _____"
        write(*,*) "|  __ \ / ____|                David Zhang"
        write(*,*) "| |__) | |     _ __ ___  ___"
        write(*,*) "|  ___/| |    | '__/ _ \/ _ \    B F G S"
        write(*,*) "| |    | |____| | |  __/ (_) |  Optimized"
        write(*,*) "|_|     \_____|_|  \___|\___/  For Spheres"
    end subroutine print_welcome_message


    subroutine initialize_point_configuration(points, energy, gradient)
        real(rk), intent(out) :: points(d + 1, num_points)
        real(rk), intent(out) :: energy, gradient(d + 1, num_points)
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
        call riesz_energy_gradient(points, energy, gradient)
        write(*,*) "Point configuration initialized."
    end subroutine initialize_point_configuration


    subroutine print_table_header
        write(*,'(A)') "#Iterations| Riesz s-energy         |&
                & RMS Gradient           | Step size"
        write(*,'(A)') "-----------+------------------------+&
                &------------------------+------------------------"
    end subroutine print_table_header


    subroutine print_optimization_status
        write(*,'(I10,A)',advance="no") iteration_count, " |"
        write(*,'(ES23.15E3,A)',advance="no") old_energy, " |"
        write(*,'(ES23.15E3,A)',advance="no") &
                & norm2(old_gradient) / sqrt(real(num_points, rk)), " |"
#ifdef PCREO_TRACK_ANGLE
        write(*,'(ES23.15E3,A)',advance="no") step_size, " |"
        write(*,'(ES23.15E3)') step_angle
#else
        write(*,'(ES23.15E3)') step_size
#endif
    end subroutine print_optimization_status


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


    real(rk) function current_time()
        integer :: ticks, tick_rate
        call system_clock(ticks, tick_rate)
        current_time = real(ticks, rk) / real(tick_rate, rk)
    end function current_time


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
                write(u,'(SP,ES23.15E3)',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end program pcreo_sphere_bfgs_hc
