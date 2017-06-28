!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo_sphere_bfgs_sym_hc - Point Configuration Riesz Energy Optimizer        !
!                            Optimized for Use on Unit Hyperspheres            !
!                            BFGS Version                                      !
!                            Symmetry Constrained                              !
!                            Hardcoded Inputs                                  !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



module constants !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The constants module contains basic constants and parameters used throughout !
! the rest of the program.                                                     !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    implicit none
    include "mkl_blas.fi"
    integer, parameter :: dp = selected_real_kind(15, 307)  ! IEEE double prec.

    real(dp), parameter :: s = 1.0d0
    integer, parameter :: d = 2
    integer, parameter :: num_points = 10

    real(dp), parameter :: print_time = 0.1d0 ! print 10 times per second
    real(dp), parameter :: save_time = 15.0d0 ! save every 15 seconds

    include "../include/icosahedral_symmetry_group.f90"
    include "../include/icosahedron_vertices.f90"

    integer, parameter :: num_vars = (d + 1) * num_points
    integer, parameter :: num_external_points = size(external_points, 2)
    integer, parameter :: symmetry_group_order = size(symmetry_group, 3)

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
        write(*,*) "Number of movable points: ", trim(adjustl(line))
        write(line,*) num_external_points
        write(*,*) "Number of fixed points: ", trim(adjustl(line))
        write(line,*) symmetry_group_order
        write(*,*) "Order of symmetry group: ", trim(adjustl(line))
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
        real(dp), intent(out) :: points(d + 1, num_points)

        real(dp) :: u(d + 1, num_points)

        call random_number(u)
        points = sqrt(-2.0d0 * log(u))
        call random_number(u)
        points = points * sin(4.0d0 * asin(1.0d0) * u)
    end subroutine random_normal_points


    pure subroutine constrain_points(points)
        real(dp), intent(inout) :: points(d + 1, num_points)

        integer :: i

        do i = 1, num_points
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points


    pure real(dp) function pair_potential(r)
        real(dp), intent(in) :: r

        pair_potential = r**(-s)
    end function pair_potential


    pure real(dp) function pair_potential_derivative(r)
        real(dp), intent(in) :: r

        pair_potential_derivative = -s * r**(-s - 1.0d0)
    end function pair_potential_derivative


    pure subroutine add_pair_energy(displ, ener)
        real(dp), intent(in) :: displ(d + 1)
        real(dp), intent(inout) :: ener

        ener = ener + pair_potential(norm2(displ))
    end subroutine add_pair_energy


    pure subroutine add_pair_energy_force(displ, ener, force)
        real(dp), intent(in) :: displ(d + 1)
        real(dp), intent(inout) :: ener, force(d + 1)

        real(dp) :: r

        r = norm2(displ)
        ener = ener + pair_potential(r)
        force = force + (pair_potential_derivative(r) / r) * displ
    end subroutine add_pair_energy_force


    pure real(dp) function riesz_energy(points) result (ener)
        real(dp), intent(in) :: points(d + 1, num_points)

        integer :: p, q, beta, delta
        real(dp) :: current_point(d + 1), displ(d + 1)

        ener = 0.0d0
        do p = 1, num_points
            do beta = 1, symmetry_group_order
                current_point = matmul(symmetry_group(:,:,beta), points(:,p))
                do q = 1, num_external_points
                    displ = current_point - external_points(:,q)
                    call add_pair_energy(displ, ener)
                end do
                do delta = 1, beta - 1
                    displ = current_point - &
                        & matmul(symmetry_group(:,:,delta), points(:,p))
                    call add_pair_energy(displ, ener)
                end do
                do delta = beta + 1, symmetry_group_order
                    displ = current_point - &
                        & matmul(symmetry_group(:,:,delta), points(:,p))
                    call add_pair_energy(displ, ener)
                end do
                do q = 1, p - 1
                    do delta = 1, symmetry_group_order
                        displ = current_point - &
                            & matmul(symmetry_group(:,:,delta), points(:,q))
                        call add_pair_energy(displ, ener)
                    end do
                end do
                do q = p + 1, num_points
                    do delta = 1, symmetry_group_order
                        displ = current_point - &
                            & matmul(symmetry_group(:,:,delta), points(:,q))
                        call add_pair_energy(displ, ener)
                    end do
                end do
            end do
        end do
        ener = 0.5d0 * ener
    end function riesz_energy


    pure subroutine riesz_energy_gradient(points, ener, grad)
        real(dp), intent(in) :: points(d + 1, num_points)
        real(dp), intent(out) :: ener, grad(d + 1, num_points)

        integer :: p, q, beta, delta
        real(dp) :: current_point(d + 1), displ(d + 1)
        real(dp) :: grad_ext(d + 1), grad_self(d + 1), grad_other(d + 1)

        ener = 0.0d0
        do p = 1, num_points
            grad(:,p) = 0.0d0
            do beta = 1, symmetry_group_order
                current_point = matmul(symmetry_group(:,:,beta), points(:,p))
                grad_ext = 0.0d0
                do q = 1, num_external_points
                    displ = current_point - external_points(:,q)
                    call add_pair_energy_force(displ, ener, grad_ext)
                end do
                grad_self = 0.0d0
                do delta = 1, beta - 1
                    displ = current_point - &
                        & matmul(symmetry_group(:,:,delta), points(:,p))
                    call add_pair_energy_force(displ, ener, grad_self)
                end do
                do delta = beta + 1, symmetry_group_order
                    displ = current_point - &
                        & matmul(symmetry_group(:,:,delta), points(:,p))
                    call add_pair_energy_force(displ, ener, grad_self)
                end do
                grad_other = 0.0d0
                do q = 1, p - 1
                    do delta = 1, symmetry_group_order
                        displ = current_point - &
                            & matmul(symmetry_group(:,:,delta), points(:,q))
                        call add_pair_energy_force(displ, ener, grad_other)
                    end do
                end do
                do q = p + 1, num_points
                    do delta = 1, symmetry_group_order
                        displ = current_point - &
                            & matmul(symmetry_group(:,:,delta), points(:,q))
                        call add_pair_energy_force(displ, ener, grad_other)
                    end do
                end do
                grad(:,p) = grad(:,p) + matmul(symmetry_group_inv(:,:,beta), &
                    & grad_ext + grad_self + grad_other)
            end do
        end do
        ener = 0.5d0 * ener
    end subroutine riesz_energy_gradient

end module sphere_riesz_energy



module bfgs_subroutines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The bfgs_subroutines module contains subroutines for the two key operations  !
! in the BFGS algorithm: line search, which in our case is a simple quadratic  !
! line search, and performing symmetric rank-two updates of the approximate    !
! inverse Hessian.                                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use ieee_arithmetic
    use sphere_riesz_energy
    implicit none

contains

    subroutine update_inverse_hessian( &
            & inv_hess, delta_gradient, step_size, step_direction)
        real(dp), intent(inout) :: &
                & inv_hess(d + 1, num_points, d + 1, num_points)
        real(dp), intent(in) :: delta_gradient(d + 1, num_points)
        real(dp), intent(in) :: step_size, step_direction(d + 1, num_points)

        real(dp) :: lambda, theta, sigma, kappa(d + 1, num_points)

        lambda = step_size * ddot(num_vars, &
                & delta_gradient, 1, step_direction, 1)
        call dsymv('U', num_vars, 1.0d0, inv_hess, &
                & num_vars, delta_gradient, 1, 0.0d0, kappa, 1)
        theta = ddot(num_vars, delta_gradient, 1, kappa, 1)
        sigma = (lambda + theta) / (lambda * lambda)
        kappa = kappa - (0.5d0 * step_size * lambda * sigma) * step_direction
        call dsyr2('U', num_vars, -step_size / lambda, &
                & kappa, 1, step_direction, 1, inv_hess, num_vars)
    end subroutine update_inverse_hessian


    include "../include/quadratic_line_search.f90"

end module bfgs_subroutines



program pcreo_sphere_bfgs_hc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Main program of pcreo_sphere_bfgs_hc. Contains subroutines for displaying    !
! and saving the current optimization status.                                  !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use bfgs_subroutines
    implicit none

    real(dp), dimension(d + 1, num_points) :: old_points, new_points
    real(dp) :: old_energy, new_energy
    real(dp), dimension(d + 1, num_points) :: old_gradient, new_gradient
    ! Approximate inverse Hessian, calculated by BFGS
    real(dp) :: inv_hess(d + 1, num_points, d + 1, num_points), step_size
    real(dp), dimension(d + 1, num_points) :: step_direction, delta_gradient
    real(dp) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

    call print_welcome_message
    write(*,*)
    call print_parameters
    write(*,*)
    call initialize_point_configuration(old_points, old_energy, old_gradient)
    write(*,*)

    ! Initialize inv_hess to identity matrix
    call dlaset('U', num_vars, num_vars, 0.0d0, 1.0d0, inv_hess, num_vars)
    ! TODO: Is there a more natural choice of initial step size?
    step_size = 1.0d-5

    iteration_count = 0
    cur_time = current_time()
    last_print_time = cur_time
    last_save_time = cur_time

    call print_table_header
    call print_optimization_status
    do
        ! Multiply inverse hessian by negative gradient to obtain step direction
        call dsymv('U', num_vars, -1.0d0, inv_hess, num_vars, &
                & old_gradient, 1, 0.0d0, step_direction, 1)
        step_size = quadratic_line_search(old_points, old_energy, &
                & step_direction, step_size)
        if (step_size == 0.0d0) then
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
        write(*,*) "| |__) | |     _ __ ___  ___    Symmetric"
        write(*,*) "|  ___/| |    | '__/ _ \/ _ \    B F G S"
        write(*,*) "| |    | |____| | |  __/ (_) |  Optimized"
        write(*,*) "|_|     \_____|_|  \___|\___/  For Spheres"
    end subroutine print_welcome_message


    subroutine initialize_point_configuration(points, energy, gradient)
        real(dp), intent(out) :: points(d + 1, num_points)
        real(dp), intent(out) :: energy, gradient(d + 1, num_points)
        integer :: u
        logical :: ex

        inquire(file="initial_reduced_configuration.txt", exist=ex)
        if (ex) then
            write(*,*) "Loading initial point configuration from file..."
            open(newunit=u, file="initial_reduced_configuration.txt")
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
                & norm2(old_gradient) / sqrt(real(num_points, dp)), " |"
        write(*,'(ES23.15E3,A)') step_size
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


    real(dp) function current_time()
        integer :: ticks, tick_rate
        call system_clock(ticks, tick_rate)
        current_time = real(ticks, dp) / real(tick_rate, dp)
    end function current_time


    subroutine save_point_file(points, idx)
        real(dp), intent(in) :: points(d + 1, num_points)
        integer, intent(in) :: idx

        integer :: i, j, u
        character(len=42) :: fname

        write(fname,"(a,i10.10,a)") "reduced_point_configuration_", idx, ".csv"
        open(file=fname, newunit=u)
        do j = 1, num_points
            do i = 1, d + 1
                if (i > 1) write(u,'(A)',advance="no") ", "
                write(u,'(ES23.15E3)',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end program pcreo_sphere_bfgs_hc
