!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! pcreo_sphere_gd_sym_hc - Point Configuration Riesz Energy Optimizer          !
!                          Optimized for Use on Unit Hyperspheres              !
!                          Gradient Descent Version                            !
!                          Symmetry Constrained                                !
!                          Hardcoded Inputs                                    !
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

    real(dp), parameter :: s = 1.0_dp
    integer, parameter :: d = 2
    integer, parameter :: num_points = 27

    real(dp), parameter :: print_time = 0.1_dp ! print 10 times per second
    real(dp), parameter :: save_time = 15.0_dp ! save every 15 seconds

    include "../include/icosahedral_symmetry_group.f90"
    include "../include/icosahedron_vertices.f90"

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
        write(line,*) external_energy
        write(*,*) "Fixed point energy: ", trim(adjustl(line))
        write(line,*) symmetry_group_order
        write(*,*) "Order of symmetry group: ", trim(adjustl(line))
        write(line,*) num_external_points + symmetry_group_order * num_points
        write(*,*) "Total number of points: ", trim(adjustl(line))
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
        points = sqrt(-2.0_dp * log(u))
        call random_number(u)
        points = points * sin(4.0_dp * asin(1.0_dp) * u)
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

        pair_potential_derivative = -s * r**(-s - 1.0_dp)
    end function pair_potential_derivative


    pure subroutine add_pair_energy(source_pt, target_pt, energy)
        real(dp), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(dp), intent(inout) :: energy

        energy = energy + pair_potential(norm2(target_pt - source_pt))
    end subroutine add_pair_energy


    pure subroutine add_pair_energy_2(source_pt, target_pt, energy)
        real(dp), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(dp), intent(inout) :: energy

        energy = energy + 2.0_dp * pair_potential(norm2(target_pt - source_pt))
    end subroutine add_pair_energy_2


    pure subroutine add_pair_energy_force(source_pt, target_pt, energy, force)
        real(dp), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(dp), intent(inout) :: energy, force(d + 1)

        real(dp) :: displacement(d + 1), r

        displacement = target_pt - source_pt
        r = norm2(displacement)
        energy = energy + pair_potential(r)
        force = force - (pair_potential_derivative(r) / r) * displacement
    end subroutine add_pair_energy_force


    pure subroutine add_pair_energy_force_2(source_pt, target_pt, energy, force)
        real(dp), intent(in) :: source_pt(d + 1), target_pt(d + 1)
        real(dp), intent(inout) :: energy, force(d + 1)

        real(dp) :: displacement(d + 1), r

        displacement = target_pt - source_pt
        r = norm2(displacement)
        energy = energy + 2.0_dp * pair_potential(r)
        force = force - (pair_potential_derivative(r) / r) * displacement
    end subroutine add_pair_energy_force_2


    pure real(dp) function riesz_energy(points)
        real(dp), intent(in) :: points(d + 1, num_points)

        integer :: b, p, q

        riesz_energy = 0.0_dp
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
            & 0.5_dp * symmetry_group_order * riesz_energy
    end function riesz_energy


    pure subroutine riesz_energy_force(points, energy, force)
        real(dp), intent(in) :: points(d + 1, num_points)
        real(dp), intent(out) :: energy, force(d + 1, num_points)

        integer :: b, p, q

        energy = 0.0_dp
        do p = 1, num_points
            force(:,p) = 0.0_dp
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
        energy = external_energy + 0.5_dp * symmetry_group_order * energy
    end subroutine riesz_energy_force

end module sphere_riesz_energy



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

    real(dp), dimension(d + 1, num_points) :: points, force
    real(dp) :: energy, step_size
    real(dp) :: last_print_time, last_save_time, cur_time
    integer :: iteration_count

    call print_welcome_message
    write(*,*)
    call print_parameters
    write(*,*)
    call initialize_point_configuration(points, energy, force)
    write(*,*)

    ! TODO: Is there a more natural choice of initial step size?
    step_size = 1.0E-10_dp

    iteration_count = 0
    cur_time = current_time()
    last_print_time = cur_time
    last_save_time = cur_time

    call print_table_header
    call print_optimization_status
    do
        step_size = quadratic_line_search(points, energy, force, step_size)
        if (step_size == 0.0_dp) then
            call print_optimization_status
            call save_point_file(points, iteration_count)
            write(*,*) "Convergence has been achieved (up to numerical&
                    & round-off error). Exiting."
            stop
        end if
        points = points + step_size * force
        call constrain_points(points)
        call riesz_energy_force(points, energy, force)
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
    end do

contains

    subroutine print_welcome_message
        write(*,*) " _____   _____"
        write(*,*) "|  __ \ / ____|                David Zhang"
        write(*,*) "| |__) | |     _ __ ___  ___    Symmetric"
        write(*,*) "|  ___/| |    | '__/ _ \/ _ \  Grad. Desc."
        write(*,*) "| |    | |____| | |  __/ (_) |  Optimized"
        write(*,*) "|_|     \_____|_|  \___|\___/  For Spheres"
    end subroutine print_welcome_message


    subroutine initialize_point_configuration(points, energy, force)
        real(dp), intent(out) :: points(d + 1, num_points)
        real(dp), intent(out) :: energy, force(d + 1, num_points)
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


    subroutine print_table_header
        write(*,'(A)') "#Iterations| Riesz s-energy         |&
                & RMS Gradient           | Step size"
        write(*,'(A)') "-----------+------------------------+&
                &------------------------+------------------------"
    end subroutine print_table_header


    subroutine print_optimization_status
        write(*,'(I10,A)',advance="no") iteration_count, " |"
        write(*,'(ES23.15E3,A)',advance="no") energy, " |"
        write(*,'(ES23.15E3,A)',advance="no") &
                & norm2(force) / sqrt(real(num_points, dp)), " |"
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
