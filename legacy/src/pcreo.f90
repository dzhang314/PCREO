!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo - Point Configuration Riesz Energy Optimizer                           !
!                                                                              !
! This program generates point configurations on (hyper)surfaces embedded in   !
! Euclidean space and optimizes their Riesz s-energy by gradient descent.      !
! For large values of s, this produces evenly distributed point                !
! configurations, while for small values of s, the points tend to cluster      !
! around the extremeties of the surface.                                       !
!                                                                              !
! The surface on which to generate point configurations is specified by giving !
! a function f for which the desired surface is the level set f(x) = 0. The    !
! gradient of f is also required. These functions should be defined in the     !
! surface_parameters module below.                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



module constants !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The constants module contains basic constants (e.g. pi, e, number kind IDs)  !
! used throughout the rest of the program.                                     !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    integer, parameter :: sp = selected_real_kind(6, 37)    ! IEEE single prec.
    integer, parameter :: dp = selected_real_kind(15, 307)  ! IEEE double prec.
    integer, parameter :: qp = selected_real_kind(33, 4931) ! IEEE quad prec.

    integer, parameter :: rk = dp ! Real kind. Edit this line to change
    ! the floating-point precision used in the rest of the program.

    real(rk), parameter :: two_pi = 6.2831853071795864769d0

end module constants



module input_parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The input_parameters module contains subroutines for retrieving and parsing  !
! command-line arguments. These arguments are stored in protected variables,   !
! so that they cannot accidentally be modified in other modules.               !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

    integer, protected :: num_points, max_iters, output_freq
    real(rk), protected :: s ! Riesz potential power parameter

contains

    integer function get_integer_argument(n) result (ans)
        integer, intent(in) :: n

        integer :: cmd_len
        character(len=:), allocatable :: cmd_arg

        call get_command_argument(n, length=cmd_len)
        allocate(character(len=cmd_len) :: cmd_arg)
        call get_command_argument(n, value=cmd_arg)
        read(cmd_arg,*) ans
        deallocate(cmd_arg)
    end function get_integer_argument


    real(rk) function get_real_argument(n) result (ans)
        integer, intent(in) :: n

        integer :: cmd_len
        character(len=:), allocatable :: cmd_arg

        call get_command_argument(n, length=cmd_len)
        allocate(character(len=cmd_len) :: cmd_arg)
        call get_command_argument(n, value=cmd_arg)
        read(cmd_arg,*) ans
        deallocate(cmd_arg)
    end function get_real_argument


    subroutine get_input_parameters
        integer :: cmd_count

        cmd_count = command_argument_count()
        if (cmd_count < 3 .or. cmd_count > 4) then
            write(*,*) "Usage: pcreo <s> <num_points>&
                                & <output_frequency> [max_iters]"
            stop
        end if

        s = get_real_argument(1)
        num_points = get_integer_argument(2)
        output_freq = get_integer_argument(3)

        if (cmd_count == 3) then
            max_iters = -1 ! if not provided, then iterate forever
        else
            max_iters = get_integer_argument(4)
        end if
    end subroutine get_input_parameters

end module input_parameters



module surface_parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The surface_parameters module contains functions and subroutines specifying  !
! the (hyper)surface on which to generate and optimize point configurations.   !
! In particular, the surface used is the zero locus of the constraint_value    !
! function.                                                                    !
!                                                                              !
! A constraint_gradient subroutine which calculates the gradient of the        !
! constraint_value function must be provided. A constraint_value_gradient      !
! subroutine, which calculates both the value and the gradient of the          !
! constraint function simultaneously, is also required. The reason is that in  !
! some cases, it may be faster to compute these quantities together (e.g. if   !
! they share common sub-expressions that only need to be computed once).       !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    implicit none

    ! Dimension of the surface and the ambient space in which it is embedded.
    ! We require srf_dim == amb_dim - 1. This may change in a later version.
    integer, parameter :: srf_dim = 2
    integer, parameter :: amb_dim = 3

    ! Bounding box for the surface. Used when generating random points.
    real(rk), parameter :: srf_min(3) = [-2.0d0, -2.0d0, -2.0d0]
    real(rk), parameter :: srf_max(3) = [+2.0d0, +2.0d0, +2.0d0]

contains

    pure real(rk) function constraint_value(x)
        real(rk), intent(in) :: x(amb_dim)

        real(rk) :: x2, y2, z2, r2, t, u, v

        x2 = x(1) * x(1)
        y2 = x(2) * x(2)
        z2 = x(3) * x(3)
        r2 = x2 + y2
        t = 1 - z2
        u = y2 - 3*x2
        v = 1 + 2*x(2)*u - 9*z2

        constraint_value = r2*r2 + v*t
    end function constraint_value


    pure subroutine constraint_gradient(x, g)
        real(rk), intent(in) :: x(amb_dim)
        real(rk), intent(out) :: g(amb_dim)

        real(rk) :: x2, y2, z2, r2, t, u, v

        x2 = x(1) * x(1)
        y2 = x(2) * x(2)
        z2 = x(3) * x(3)
        r2 = x2 + y2
        t = 1 - z2
        u = y2 - 3*x2
        v = 1 + 2*u*x(2) - 9*z2

        g(1) = 4*x(1)*(r2 - 3*x(2)*t)
        g(2) = 4*x(2)*r2 - 6*t*(x2 - y2)
        g(3) = -2*x(3)*(v + 9*t)
    end subroutine constraint_gradient


    pure subroutine constraint_value_gradient(x, c, g)
        real(rk), intent(in) :: x(amb_dim)
        real(rk), intent(out) :: c, g(amb_dim)

        real(rk) :: x2, y2, z2, r2, t, u, v

        x2 = x(1) * x(1)
        y2 = x(2) * x(2)
        z2 = x(3) * x(3)
        r2 = x2 + y2
        t = 1 - z2
        u = y2 - 3*x2
        v = 1 + 2*u*x(2) - 9*z2

        c = r2*r2 + v*t
        g(1) = 4*x(1)*(r2 - 3*x(2)*t)
        g(2) = 4*x(2)*r2 - 6*t*(x2 - y2)
        g(3) = -2*x(3)*(v + 9*t)
    end subroutine constraint_value_gradient

end module surface_parameters



module constrained_energy_optimization

    use ieee_arithmetic
    use surface_parameters
    implicit none

contains

    pure subroutine constrain(x, c)
        real(rk), intent(inout) :: x(amb_dim)
        real(rk), intent(out) :: c

        real(rk) :: c_old, c_new, alpha
        real(rk) :: g(amb_dim) ! constraint gradient at original point
        real(rk) :: h(amb_dim) ! constraint gradient at most recent point
        real(rk) :: y(amb_dim) ! last known good point
        real(rk) :: z(amb_dim) ! most recent point

        y = x
        call constraint_value_gradient(x, c_old, g)
        alpha = -c_old / dot_product(g, g)
        do
            z = x + alpha * g
            call constraint_value_gradient(z, c_new, h)
            if (ieee_is_finite(c_new) .and. ieee_is_finite(c_old) .and. &
                & abs(c_new) < abs(c_old)) then
                alpha = alpha - c_new / dot_product(g, h)
                c_old = c_new
                y = z
            else ! Improvement has stopped. End Newton iteration.
                exit
            end if
        end do
        x = y
        c = c_old
    end subroutine constrain


    pure subroutine constrain_points(points, c_values)
        real(rk), intent(inout) :: points(:,:)
        real(rk), intent(out) :: c_values(:)

        integer :: i

        do i = 1, size(points, 2)
            call constrain(points(:,i), c_values(i))
        end do
    end subroutine constrain_points


    subroutine random_surface_point(x, c)
        real(rk), intent(out) :: x(amb_dim), c

        integer :: i

        do
            do i = 1, amb_dim
                call random_number(x(i))
                x(i) = srf_min(i) + (srf_max(i) - srf_min(i)) * x(i)
            end do
            call constrain(x, c)
            if (abs(constraint_value(x)) < 10*epsilon(x)) return
        end do
    end subroutine random_surface_point


    subroutine random_surface_points(points, c_values)
        real(rk), intent(out) :: points(:,:), c_values(:)

        integer :: i

        do i = 1, size(points, 2)
            call random_surface_point(points(:,i), c_values(i))
        end do
    end subroutine random_surface_points


    pure subroutine tangential_component(x, p)
        real(rk), intent(in) :: x(amb_dim)
        real(rk), intent(inout) :: p(amb_dim)

        real(rk) :: g(amb_dim), alpha

        call constraint_gradient(x, g)
        alpha = dot_product(p, g) / dot_product(g, g)
        p = p - alpha * g
    end subroutine tangential_component


    pure real(rk) function riesz_energy(s, points)
        real(rk), intent(in) :: s, points(:,:)

        integer :: i, j

        riesz_energy = 0
        do j = 1, size(points, 2)
            do i = 1, j - 1
                riesz_energy = riesz_energy + &
                        & norm2(points(:,i) - points(:,j))**(-s)
            end do
        end do
    end function riesz_energy


    pure subroutine riesz_energy_gradient(s, points, ener, grad)
        real(rk), intent(in) :: s, points(:,:)
        real(rk), intent(out) :: ener, grad(:,:)

        real(rk) :: displ(amb_dim), dist_sq, term
        integer :: i, j

        ener = 0
        do j = 1, size(points, 2)
            grad(:,j) = 0
            do i = 1, j - 1
                displ = points(:,i) - points(:,j)
                dist_sq = dot_product(displ, displ)
                term = dist_sq**(-0.5d0 * s)
                ener = ener + term
                term = s * term / dist_sq
                grad(:,j) = grad(:,j) + term * displ
            end do
            do i = j + 1, size(points, 2)
                displ = points(:,i) - points(:,j)
                term = s * norm2(displ)**(-s - 2.0d0)
                grad(:,j) = grad(:,j) + term * displ
            end do
            call tangential_component(points(:,j), grad(:,j))
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
        real(rk), intent(in) :: step(:,:)

        step_norm = norm2(step)
    end function step_norm

end module constrained_energy_optimization



program test

    use input_parameters
    use constrained_energy_optimization
    implicit none

    real(rk), allocatable :: points_old(:,:)
    real(rk), allocatable :: points_new(:,:)
    real(rk), allocatable :: grad_step(:,:)
    real(rk), allocatable :: c_values(:)

    real(rk) :: ener_old, ener_new, step_size, grad_norm, constraint_err
    integer :: u, num_good_iters, num_bad_iters
    character(len=34) :: fname

    call init_random_seed
    call print_welcome_message
    call get_input_parameters
    call print_input_parameters
    write(*,*)

    allocate(points_old(amb_dim, num_points))
    allocate(points_new(amb_dim, num_points))
    allocate(grad_step(amb_dim, num_points))
    allocate(c_values(num_points))

    write(*,*) "Generating random initial point configuration..."
    call random_surface_points(points_old, c_values)
    call riesz_energy_gradient(s, points_old, ener_old, grad_step)
    write(*,*) "Point configuration initialized."
    write(*,*)
    write(*,*) "# Good steps| # Bad steps | Riesz s-energy         &
            & | Gradient norm           | Step size              &
            & | Constraint error"
    write(*,*) "------------+-------------+------------------------&
            &-+-------------------------+------------------------&
            &-+-------------------------"

    step_size = 1
    num_good_iters = 0
    num_bad_iters = 0
    do
        grad_norm = step_norm(grad_step)
        grad_step = step_size * grad_step / grad_norm
        points_new = points_old - grad_step
        call constrain_points(points_new, c_values)
        constraint_err = maxval(abs(c_values))
        if (constraint_err > 1.0d-10) then
            step_size = 1.0d-2 * step_size
            num_bad_iters = num_bad_iters + 1
            cycle
        end if
        call riesz_energy_gradient(s, points_new, ener_new, grad_step)
        if (ener_new < ener_old) then
            num_good_iters = num_good_iters + 1
            write(*,*) num_good_iters, "|", num_bad_iters, "|", &
                    & ener_new, "|", grad_norm, "|", step_size, "|", &
                    & constraint_err
            if (mod(num_good_iters, output_freq) == 0 .or. &
                    & num_good_iters == max_iters) then
                write(fname,"(a,i10.10,a)") "point_configuration_", &
                        & num_good_iters, ".txt"
                open(file=fname, newunit=u)
                write(u,*) points_new
                close(u)
            end if
            if (num_good_iters == max_iters) exit
            points_old = points_new
            ener_old = ener_new
        else
            num_bad_iters = num_bad_iters + 1
            step_size = 0.5d0 * step_size
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
        write(*,*) "Value of s (Riesz potential power parameter): ", s
        write(*,*) "Number of points: ", num_points
        write(*,*) "Output frequency: every ", output_freq, " iterations"

        if (max_iters == -1) then
            write(*,*) "Maximum number of iterations not specified.&
                        & Iterating indefinitely."
        else
            write(*,*) "Maximum number of iterations: ", max_iters
        end if
    end subroutine print_input_parameters


    subroutine print_welcome_message
        write(*,*) "  _____   _____                             "
        write(*,*) " |  __ \ / ____|                Version 1.0 "
        write(*,*) " | |__) | |     _ __ ___  ___               "
        write(*,*) " |  ___/| |    | '__/ _ \/ _ \  David Zhang "
        write(*,*) " | |    | |____| | |  __/ (_) |             "
        write(*,*) " |_|     \_____|_|  \___|\___/  TDT Surface "
        write(*,*) "                                            "
    end subroutine print_welcome_message

end program test
