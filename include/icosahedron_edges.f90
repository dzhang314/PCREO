    real(rk), parameter :: external_points(3, 30) = reshape((/ &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +1.0000000000000000000000000000000000000000E+0000_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & -1.0000000000000000000000000000000000000000E+0000_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +1.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & -1.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & +8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & +5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & +3.0901699437494742410229341718281905886015E-0001_rk, &
        & -8.0901699437494742410229341718281905886015E-0001_rk, &
        & -5.0000000000000000000000000000000000000000E-0001_rk, &
        & -3.0901699437494742410229341718281905886015E-0001_rk, &
        & +1.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & -1.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk, &
        & +0.0000000000000000000000000000000000000000E+0000_rk /), &
        & (/ 3, 30 /))

    real(rk), parameter :: external_energy = &
        & 60.0_rk / 6.1803398874989484820458683436563811772030E-0001_rk**s + &
        & 60.0_rk / 1.0000000000000000000000000000000000000000E+0000_rk**s + &
        & 60.0_rk / 1.1755705045849462583374119092781455371953E+0000_rk**s + &
        & 60.0_rk / 1.4142135623730950488016887242096980785696E+0000_rk**s + &
        & 60.0_rk / 1.6180339887498948482045868343656381177203E+0000_rk**s + &
        & 60.0_rk / 1.7320508075688772935274463415058723669428E+0000_rk**s + &
        & 60.0_rk / 1.9021130325903071442328786667587642868113E+0000_rk**s + &
        & 15.0_rk / 2.0000000000000000000000000000000000000000E+0000_rk**s