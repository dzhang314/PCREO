!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo_sphere_bfgs_hc - Point Configuration Riesz Energy Optimizer            !
!                        Optimized for Use on Unit Hyperspheres                !
!                        BFGS Version                                          !
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
    integer, parameter :: d = 3

    real(dp), parameter :: print_time = 0.1d0 ! print 10 times per second
    real(dp), parameter :: save_time = 15.0d0 ! save every 15 seconds

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

    integer, protected :: num_points, num_vars

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


    real(dp) function get_real_argument(n) result (ans)
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
        if (cmd_count /= 1) then
            write(*,*) "Usage: pcreo_sphere_bfgs <num_points>"
            stop
        end if

        num_points = get_integer_argument(1)
        num_vars = (d + 1) * num_points
    end subroutine get_input_parameters


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

end module input_parameters



module sphere_riesz_energy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The sphere_riesz_energy module contains subroutines for generating random    !
! point configurations on the unit (hyper)sphere and computing their Riesz     !
! s-energy.                                                                    !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use input_parameters
    implicit none

contains

    subroutine random_normal_points(points)
        real(dp), intent(out) :: points(:,:)

        real(dp), allocatable :: u(:,:)

        allocate(u, mold=points)
        call random_number(u)
        points = sqrt(-2.0d0 * log(u))
        call random_number(u)
        points = points * sin(4.0d0 * asin(1.0d0) * u)
    end subroutine random_normal_points


    pure subroutine constrain_points(points)
        real(dp), intent(inout) :: points(:,:)

        integer :: i

        do i = 1, size(points, 2)
            points(:,i) = points(:,i) / norm2(points(:,i))
        end do
    end subroutine constrain_points


    pure real(dp) function riesz_energy(points)
        real(dp), intent(in) :: points(:,:)

        integer :: i, j

        riesz_energy = 0.0d0
        do j = 1, size(points, 2)
            do i = 1, j - 1
                riesz_energy = riesz_energy + &
                        & norm2(points(:,i) - points(:,j))**(-s)
            end do
        end do
    end function riesz_energy


    pure subroutine riesz_energy_gradient(points, ener, grad)
        real(dp), intent(in) :: points(:,:)
        real(dp), intent(out) :: ener, grad(:,:)

        real(dp) :: displ(d + 1), dist_sq, term
        integer :: i, j

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
            grad(:,j) = grad(:,j) - &
                & dot_product(grad(:,j), points(:,j)) * points(:,j)
        end do
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
    use input_parameters
    use sphere_riesz_energy
    implicit none

contains

    subroutine update_inverse_hessian( &
            & inv_hess, delta_gradient, step_size, step_direction)
        real(dp), intent(inout) :: inv_hess(:,:,:,:)
        real(dp), intent(in) :: delta_gradient(:,:)
        real(dp), intent(in) :: step_size, step_direction(:,:)

        real(dp) :: lambda, theta, sigma
        real(dp), allocatable :: kappa(:,:)

        lambda = step_size * ddot(num_vars, &
                & delta_gradient, 1, step_direction, 1)
        allocate(kappa(d + 1, num_points))
        call dsymv('U', num_vars, 1.0d0, inv_hess, &
                & num_vars, delta_gradient, 1, 0.0d0, kappa, 1)
        theta = ddot(num_vars, delta_gradient, 1, kappa, 1)
        sigma = (lambda + theta) / (lambda * lambda)
        kappa = kappa - (0.5d0 * step_size * lambda * sigma) * step_direction
        call dsyr2('U', num_vars, -step_size / lambda, &
                & kappa, 1, step_direction, 1, inv_hess, num_vars)
    end subroutine update_inverse_hessian


    pure real(dp) function quadratic_line_search(points, energy, &
            & step_direction, initial_step_size) result (optimal_step_size)
        real(dp), intent(in) :: points(:,:)
        real(dp), intent(in) :: energy
        real(dp), intent(in) :: step_direction(:,:)
        real(dp), intent(in) :: initial_step_size

        real(dp), allocatable :: new_points(:,:), newer_points(:,:)
        real(dp) :: step_size, new_energy, newer_energy
        integer :: num_increases

        ! The goal of quadratic line search is to find three points, a, b, c,
        ! such that a < b < c and f(a) > f(b) < f(c). We say that such a
        ! triplet of points is "bowl-shaped." Once we have three bowl-shaped
        ! points, we fit a parabola through them, and take its minimum as our
        ! best step size. In this implementation, we take a = 0, so that f(a)
        ! is the energy at the initial point, and work to find b and c such
        ! that c = 2*b.
        
        allocate(new_points, newer_points, mold=points)

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
                newer_points = points + (2.0d0 * step_size) * step_direction
                call constrain_points(newer_points)
                newer_energy = riesz_energy(newer_points)
                ! If so, then we have our bowl-shaped points, and we exit
                ! the loop.
                if (newer_energy >= new_energy) then
                    exit
                ! If not, then we can be even more ambitious. Double the
                ! step size again.
                else
                    step_size = 2.0d0 * step_size
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
            optimal_step_size = 0.5d0 * step_size * &
                    & (4.0d0 * new_energy - newer_energy - 3.0d0 * energy) / &
                    & (2.0d0 * new_energy - newer_energy - energy)
            ! Note that this formula is numerically unstable, since it contains
            ! subtractions of roughly equal-magnitude numbers that can result
            ! in catastrophic cancellation. To check whether this has occurred,
            ! we perform one last sanity check: the arg min should fall
            ! somewhere inside the bowl.
            if (0.0d0 < optimal_step_size .and. &
                    & optimal_step_size < 2.0d0 * step_size) then
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
                newer_points = points + (0.5d0 * step_size) * step_direction
                call constrain_points(newer_points)
                newer_energy = riesz_energy(newer_points)
                ! If so, then we have our bowl-shaped points, and we exit
                ! the loop.
                if (newer_energy < energy) then
                    exit
                ! Otherwise, we need to halve the step size and try again.
                else
                    step_size = 0.5d0 * step_size
                    ! If no step produces a decrease, no matter how small, then
                    ! we have probably started our search from a local minimum.
                    ! Return zero step size to indicate this.
                    if (step_size == 0.0d0) then
                        optimal_step_size = 0.0d0
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
            optimal_step_size = 0.25d0 * step_size * &
                    & (new_energy - 4.0d0 * newer_energy + 3.0d0 * energy) / &
                    & (new_energy - 2.0d0 * newer_energy + energy)
            ! We perform a similar sanity check to guard against numerical
            ! instability.
            if (0.0d0 < optimal_step_size .and. &
                    & optimal_step_size < step_size) then
                return
            else
                optimal_step_size = 0.5d0 * step_size
                return
            end if
        end if
    end function quadratic_line_search

end module bfgs_subroutines



program pcreo_sphere_bfgs_hc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Main program of pcreo_sphere_bfgs_hc. Contains subroutines for displaying    !
! and saving the current optimization status.                                  !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use input_parameters
    use sphere_riesz_energy
    use bfgs_subroutines
    implicit none

    real(dp), allocatable :: old_points(:,:), new_points(:,:)
    real(dp) :: old_energy, new_energy
    real(dp), allocatable :: old_gradient(:,:), new_gradient(:,:)
    ! Angle between old and new gradient
    real(dp) :: gradient_angle
    ! Approximate inverse Hessian, calculated by BFGS
    real(dp), allocatable :: inv_hess(:,:,:,:)
    real(dp) :: step_size
    real(dp), allocatable :: step_direction(:,:), delta_gradient(:,:)
    real(dp) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

    call print_welcome_message
    call get_input_parameters
    write(*,*)
    call print_parameters
    write(*,*)

    allocate(old_points(d + 1, num_points), new_points(d + 1, num_points))
    allocate(old_gradient(d + 1, num_points), new_gradient(d + 1, num_points))
    allocate(inv_hess(d + 1, num_points, d + 1, num_points))
    allocate(step_direction(d + 1, num_points), delta_gradient(d + 1, num_points))

    call initialize_point_configuration(old_points, old_energy, old_gradient)
    write(*,*)

    ! Initialize inv_hess to identity matrix
    call dlaset('U', num_vars, num_vars, 0.0d0, 1.0d0, inv_hess, num_vars)
    ! TODO: Is there a more natural choice of initial step size?
    step_size = 1.0d-10

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
        gradient_angle = sum(old_gradient * new_gradient) / &
            & (norm2(old_gradient) * norm2(new_gradient))
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
        real(dp), intent(out) :: points(d + 1, num_points)
        real(dp), intent(out) :: energy, gradient(d + 1, num_points)
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
                & norm2(old_gradient) / sqrt(real(num_points, dp)), " |"
        write(*,'(ES23.15E3,A)',advance="no") step_size, " |"
        write(*,'(ES23.15E3)') gradient_angle
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
