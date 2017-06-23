!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! pcreo_sphere - Point Configuration Riesz Energy Optimizer                    !
!                Optimized for Use on Unit Hyperspheres                        !
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

    integer, protected :: num_points, max_iters, output_freq, sph_dim, amb_dim
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
        if (cmd_count < 4 .or. cmd_count > 5) then
            write(*,*) "Usage: pcreo_sphere <sph_dim> <s> <num_points>&
                                & <output_frequency> [max_iters]"
            stop
        end if

        sph_dim = get_integer_argument(1)
        amb_dim = sph_dim + 1
        s = get_real_argument(2)
        num_points = get_integer_argument(3)
        output_freq = get_integer_argument(4)

        if (cmd_count == 4) then
            max_iters = -1 ! if not provided, then iterate forever
        else
            max_iters = get_integer_argument(5)
        end if
    end subroutine get_input_parameters

end module input_parameters



module sphere_energy_optimization

    use constants
    use input_parameters
    implicit none

contains

    pure subroutine constrain_points(points)
        real(rk), intent(inout) :: points(:,:)

        integer :: i

        do i = 1, size(points, 2)
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points


    real(rk) function random_normal()
        real(rk), save :: u
        real(rk) :: r, theta
        logical, save :: computed = .false.

        if (computed) then
            random_normal = u
            computed = .false.
        else
            call random_number(u)
            r = sqrt(-2.0d0 * log(u))
            call random_number(u)
            theta = two_pi * u
            random_normal = r * cos(theta)
            u = r * sin(theta)
            computed = .true.
        end if
    end function random_normal


    subroutine random_sphere_points(points)
        real(rk), intent(out) :: points(:,:)

        integer :: i, j

        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                points(i,j) = random_normal()
            end do
        end do
        call constrain_points(points)
    end subroutine random_sphere_points


    subroutine riesz_energy_gradient(s, points, ener, grad)
        real(rk), intent(in) :: s, points(:,:)
        real(rk), intent(out) :: ener, grad(:,:)

        real(rk), allocatable, save :: displ(:)
        real(rk) :: dist_sq, term
        integer :: i, j

        if (.not. allocated(displ)) allocate(displ(amb_dim))

        ener = 0.0d0
        do j = 1, size(points, 2)
            grad(:,j) = 0.0d0
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
        real(rk), intent(in) :: step(:,:)

        step_norm = norm2(step)
    end function step_norm

end module sphere_energy_optimization



program test

    use input_parameters
    use sphere_energy_optimization
    implicit none

    real(rk), allocatable :: points_old(:,:)
    real(rk), allocatable :: points_new(:,:)
    real(rk), allocatable :: grad_step(:,:)

    real(rk) :: ener_old, ener_new, step_size, grad_norm
    integer :: num_good_iters, num_bad_iters

    call init_random_seed
    call print_welcome_message
    call get_input_parameters
    call print_input_parameters
    write(*,*)

    write(*,*) "Generating random initial point configuration..."
    allocate(points_old(amb_dim, num_points))
    allocate(points_new(amb_dim, num_points))
    allocate(grad_step(amb_dim, num_points))
    call random_sphere_points(points_old)
    call riesz_energy_gradient(s, points_old, ener_old, grad_step)
    write(*,*) "Point configuration initialized."
    write(*,*)
    write(*,*) "# Good steps| # Bad steps | Riesz s-energy           &
            & | Gradient norm             | Step size"
    write(*,*) "------------+-------------+--------------------------&
            &-+---------------------------+---------------------------"

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
            if (mod(num_good_iters, output_freq) == 0 .or. &
                    & num_good_iters == max_iters) then
                call save_point_file(points_new, num_good_iters)
            end if
            if (num_good_iters == max_iters) exit
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
        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                if (i > 1) write(u,'(A)',advance="no") ", "
                write(u,'(E24.16E3)',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end program test
