!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! pcreo_sphere_gd_hc - Point Configuration Riesz Energy Optimizer              !
!                      Optimized for Use on Unit Hyperspheres                  !
!                      Gradient Descent Version                                !
!                      Hardcoded Inputs                                        !
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
! Note that this program does not accept command-line arguments. Instead,      !
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
    integer, parameter :: dp = selected_real_kind(15, 307)  ! IEEE double prec.

    real(dp), parameter :: s = 1.0d0
    integer, parameter :: d = 2
    integer, parameter :: num_points = 1632

    real, parameter :: print_time = 0.1 ! print 10 times per second
    real, parameter :: save_time = 15.0 ! save every 15 seconds

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


    pure real(dp) function riesz_energy(points)
        real(dp), intent(in) :: points(d + 1, num_points)

        integer :: i, j

        riesz_energy = 0.0d0
        do j = 1, num_points
            do i = 1, j - 1
                riesz_energy = riesz_energy + &
                        & norm2(points(:,i) - points(:,j))**(-s)
            end do
        end do
    end function riesz_energy


    pure subroutine riesz_energy_gradient(points, ener, grad)
        real(dp), intent(in) :: points(d + 1, num_points)
        real(dp), intent(out) :: ener, grad(d + 1, num_points)

        real(dp) :: displ(d + 1), dist_sq, term
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
            grad(:,j) = grad(:,j) - &
                & dot_product(grad(:,j), points(:,j)) * points(:,j)
        end do
    end subroutine riesz_energy_gradient

end module sphere_riesz_energy



module line_search !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! The line_search module contains a single subroutine implementing a simple    !
! quadratic line search algorithm.                                             !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use ieee_arithmetic
    use sphere_riesz_energy
    implicit none

contains

    pure real(dp) function quadratic_line_search(points, energy, &
            & step_direction, initial_step_size) result (optimal_step_size)
        real(dp), intent(in) :: points(d + 1, num_points)
        real(dp), intent(in) :: energy
        real(dp), intent(in) :: step_direction(d + 1, num_points)
        real(dp), intent(in) :: initial_step_size

        real(dp), dimension(d + 1, num_points) :: new_points, newer_points
        real(dp) :: step_size, new_energy, newer_energy
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

        ! If the new energy is greater than the old energy, or worse, gives
        ! a non-finite (Inf/NaN) result, then we know our initial step size
        ! was too large.
        if (new_energy > energy .or. .not. ieee_is_finite(new_energy)) then
            do
                ! Try taking a step of half the size. Does this result in a
                ! decrease?
                newer_points = points + (0.5d0 * step_size) * step_direction
                call constrain_points(newer_points)
                newer_energy = riesz_energy(newer_points)
                ! If so, then we have our bowl-shaped points, and we exit
                ! the loop.
                if (newer_energy <= energy) then
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
            ! Finally, once we have our bowl-shaped points, we take the arg
            ! min of the interpolating parabola. The formula, worked out in
            ! advance, is as follows:
            optimal_step_size = 0.25d0 * step_size * &
                    & (new_energy - 4.0d0 * newer_energy + 3.0d0 * energy) / &
                    & (new_energy - 2.0d0 * newer_energy + energy)
            ! Note that this formula is numerically unstable, since it contains
            ! subtractions of roughly equal-magnitude numbers that can result
            ! in catastrophic cancellation. To check whether this has occurred,
            ! we perform one last sanity check: the arg min should fall
            ! somewhere inside the bowl.
            if (0.0d0 < optimal_step_size .and. &
                    & optimal_step_size < step_size) then
                return
            ! If our sanity check has failed, then the bowl we found must be so
            ! shallow that it doesn't really matter what step size we return.
            ! Just take the middle of the bowl.
            else
                optimal_step_size = 0.5d0 * step_size
                return
            end if
        ! Now, if the new energy is less than the old energy, then we can
        ! afford to be a bit more ambitious. We try a larger step size.
        else if (new_energy < energy) then
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
                ! If not, then we need to be even more ambitious. Double the
                ! step size again.
                else
                    step_size = 2.0d0 * step_size
                    new_points = newer_points
                    new_energy = newer_energy
                    num_increases = num_increases + 1
                    ! We might run into a situation where increasing the step
                    ! size "accidentally" decreases the energy by, say, jumping
                    ! into the basin of a deeper local minimum. To prevent this
                    ! from getting us too far off track, we cap the number of
                    ! consecutive times the step size can increase.
                    if (num_increases >= 4) then
                        optimal_step_size = step_size
                        return
                    end if
                end if
            end do
            ! Again, we use the following formula for the arg min of the
            ! interpolating parabola. Note that this is slightly different than
            ! the previous one -- here we have b = step_size, c = 2*step_size,
            ! whereas before we had b = step_size/2, c = step_size.
            optimal_step_size = 0.5d0 * step_size * &
                    & (4.0d0 * new_energy - newer_energy - 3.0d0 * energy) / &
                    & (2.0d0 * new_energy - newer_energy - energy)
            ! We perform a similar sanity check to guard against numerical
            ! instability.
            if (0.0d0 < optimal_step_size .and. &
                    & optimal_step_size < step_size) then
                return
            else
                optimal_step_size = 0.5d0 * step_size
                return
            end if
        ! If our initial step produces exactly zero change in energy, then we
        ! are in a very exceptional situation. Most likely, either the supplied
        ! energy function is actually constant, or we were given an identically
        ! zero step direction. Just return half the initial step size and see
        ! what happens.
        else
            optimal_step_size = 0.5d0 * step_size
        end if
    end function quadratic_line_search

end module line_search



program pcreo_sphere_gd_hc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Main program of pcreo_sphere_gd_hc. Contains subroutines for displaying and  !
! saving the current optimization status.                                      !
!                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    use constants
    use sphere_riesz_energy
    use line_search
    implicit none

    real(dp), dimension(d + 1, num_points) :: old_points, new_points
    real(dp), dimension(d + 1, num_points) :: old_gradient, new_gradient
    real(dp) :: old_energy, new_energy, step_size
    real(dp) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

    call print_welcome_message
    write(*,*)
    call print_parameters
    write(*,*)
    call initialize_point_configuration(old_points, old_energy, old_gradient)
    write(*,*)

    ! TODO: Is there a more natural choice of initial step size?
    step_size = 1.0d-5

    iteration_count = 0
    cur_time = current_time()
    last_print_time = cur_time
    last_save_time = cur_time

    call print_table_header
    call print_optimization_status
    do
        step_size = quadratic_line_search(old_points, old_energy, &
                & old_gradient, step_size)
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
    end


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
                write(u,'(ES23.15E3)',advance="no") points(i,j)
            end do
            write(u,*)
        end do
        close(u)
    end subroutine save_point_file

end program pcreo_sphere_gd_hc
