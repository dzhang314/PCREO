!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! pcreo_sphere_hc - Point Configuration Riesz Energy Optimizer                 !
!                   Optimized for Use on Unit Hyperspheres                     !
!                   Hardcoded Inputs                                           !
!                                                                              !
! This program generates point configurations on the unit d-sphere embedded in !
! (d+1)-dimensional Euclidean space and optimizes their Riesz s-energy by      !
! gradient descent. It differs from the generic version of pcreo in that       !
! certain structural optimizations have been performed which only make sense   !
! for unit (hyper)spheres.                                                     !
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
! Note that this program does not accept command-line arguments. Instead,      !
! all parameters (number of points, value of s, etc.) have been made           !
! compile-time constants, hardcoded in the constants module below. This        !
! provides a significant speed boost with the Intel Fortran compiler, at the   !
! inconvenience of having to re-compile for every set of parameters.           !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



module sphere_energy_optimization

    implicit none
    integer, parameter :: sp = selected_real_kind(6, 37)    ! IEEE single prec.
    integer, parameter :: dp = selected_real_kind(15, 307)  ! IEEE double prec.
    integer, parameter :: qp = selected_real_kind(33, 4931) ! IEEE quad prec.

    integer, parameter :: rk = dp ! Real kind. Edit this line to change
    ! the floating-point precision used in the rest of the program.

    real(rk), parameter :: two_pi = 6.2831853071795864769d0

    integer, parameter :: num_points = 1000
    integer, parameter :: output_freq = 50
    real(rk), parameter :: s = 2.0d0 ! Riesz potential power parameter
    integer, parameter :: sph_dim = 3
    integer, parameter :: amb_dim = sph_dim + 1

contains

    pure subroutine constrain_points(points)
        real(rk), intent(inout) :: points(amb_dim, num_points)

        integer :: i

        do i = 1, num_points
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points


    subroutine random_sphere_points(points)
        real(rk), intent(out) :: points(amb_dim, num_points)

        real(rk) :: u(amb_dim, num_points)

        call random_number(u)
        points = sqrt(-2.0d0 * log(u))
        call random_number(u)
        points = points * sin(two_pi * u)
        call constrain_points(points)
    end subroutine random_sphere_points


    pure subroutine riesz_energy_gradient(s, points, ener, grad)
        real(rk), intent(in) :: s, points(amb_dim, num_points)
        real(rk), intent(out) :: ener, grad(amb_dim, num_points)

        real(rk) :: displ(amb_dim), dist_sq, term
        integer :: i, j

        ener = 0.0d0
        do j = 1, num_points
            grad(:,j) = 0.0d0
            do i = 1, j - 1
                displ = points(:,i) - points(:,j)
                dist_sq = dot_product(displ, displ)
                term = dist_sq**(-0.5d0 * s)
                ener = ener + term
                term = s * term / dist_sq
                grad(:,j) = grad(:,j) + term * displ
            end do
            do i = j + 1, num_points
                displ = points(:,i) - points(:,j)
                term = s * norm2(displ)**(-s - 2.0d0)
                grad(:,j) = grad(:,j) + term * displ
            end do
            grad(:,j) = grad(:,j) - (dot_product(grad(:,j), points(:,j)) / &
                    & dot_product(points(:,j), points(:,j))) * points(:,j)
        end do
    end subroutine riesz_energy_gradient


!~     pure real(rk) function step_norm(step)
!~         real(rk), intent(in) :: step(:,:)

!~         real(rk) :: norm_sq
!~         integer :: i

!~         step_norm = 0
!~         do i = 1, num_points
!~             norm_sq = step(1,i)*step(1,i) + step(2,i)*step(2,i) + step(3,i)*step(3,i)
!~             if (norm_sq > step_norm) step_norm = norm_sq
!~         end do
!~         step_norm = sqrt(step_norm)
!~     end function step_norm


    pure real(rk) function step_norm(step)
        real(rk), intent(in) :: step(amb_dim, num_points)

        step_norm = norm2(step)
    end function step_norm

end module sphere_energy_optimization



program test

    use sphere_energy_optimization
    implicit none

    real(rk) :: points_old(amb_dim, num_points)
    real(rk) :: points_new(amb_dim, num_points)
    real(rk) :: grad_step(amb_dim, num_points)

    real(rk) :: ener_old, ener_new, step_size, grad_norm
    integer :: num_good_iters, num_bad_iters

    call init_random_seed
    call print_welcome_message
    call print_input_parameters
    write(*,*)

    write(*,*) "Generating random initial point configuration..."
    call random_sphere_points(points_old)
    call riesz_energy_gradient(s, points_old, ener_old, grad_step)
    write(*,*) "Point configuration initialized."
    write(*,*)
    write(*,*) "# Good steps| # Bad steps | Riesz s-energy         &
            & | Gradient norm           | Step size"
    write(*,*) "------------+-------------+------------------------&
            &-+-------------------------+-------------------------"

    step_size = 1
    num_good_iters = 0
    num_bad_iters = 0
    do
        grad_norm = step_norm(grad_step)
        grad_step = step_size * grad_step / grad_norm
        points_new = points_old - grad_step
        call constrain_points(points_new)
        call riesz_energy_gradient(s, points_new, ener_new, grad_step)
        if (ener_new < ener_old) then
            num_good_iters = num_good_iters + 1
            write(*,*) num_good_iters, "|", num_bad_iters, "|", &
                    & ener_new, "|", grad_norm, "|", step_size
            if (mod(num_good_iters, output_freq) == 0) then
                call save_point_file(points_new, num_good_iters)
            end if
            points_old = points_new
            ener_old = ener_new
        else
            num_bad_iters = num_bad_iters + 1
            step_size = 0.5d0 * step_size
            if (step_size < epsilon(step_size)) then
                write(*,*) "Convergence has been achieved (up to numerical&
                        & round-off error). Exiting."
                call save_point_file(points_old, num_good_iters)
                stop
            end if
        end if
    end do

contains

    subroutine init_random_seed
        integer :: i, n, clock
        integer, allocatable :: seed(:)

        call random_seed(size=n)
        allocate(seed(n))
        call system_clock(count=clock)
        seed = clock + 37 * (/ (i, i = 1,n) /)
        call random_seed(put=seed)
        deallocate(seed)
    end subroutine init_random_seed


    subroutine print_input_parameters
        write(*,*) "Sphere dimension: ", sph_dim
        write(*,*) "Embedded in dimension: ", amb_dim
        write(*,*) "Value of s (Riesz potential power parameter): ", s
        write(*,*) "Number of points: ", num_points
        write(*,*) "Output frequency: every ", output_freq, " iterations"
    end subroutine print_input_parameters


    subroutine print_welcome_message
        write(*,*) "  _____   _____                             "
        write(*,*) " |  __ \ / ____|                Version 1.0 "
        write(*,*) " | |__) | |     _ __ ___  ___   David Zhang "
        write(*,*) " |  ___/| |    | '__/ _ \/ _ \              "
        write(*,*) " | |    | |____| | |  __/ (_) |  Optimized  "
        write(*,*) " |_|     \_____|_|  \___|\___/  For Spheres "
        write(*,*) "                                            "
    end subroutine print_welcome_message


    subroutine save_point_file(points, idx)
        real(rk), intent(in) :: points(:,:)
        integer, intent(in) :: idx

        integer :: i, j, u
        character(len=34) :: fname

        write(fname,"(a,i10.10,a)") "point_configuration_", idx, ".csv"
        open(file=fname, newunit=u)
        do j = 1, num_points
            do i = 1, amb_dim
                if (i > 1) write(u,'(A)',advance="no") ", "
                write(u,'(E24.16E3)',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end program test
