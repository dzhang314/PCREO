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
                    if (step_size == 0.0_rk) then
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
