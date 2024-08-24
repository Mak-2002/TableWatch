<?php

namespace App\Http\Controllers\Admin\Reservation;

use App\Http\Controllers\Controller;
use App\Models\Reservation;
use Illuminate\Http\Request;

class ReservationController extends Controller
{
    //

    public function index()
    {
        $reservations = Reservation::all();
        return view('Admin.Reservation.index' , compact('reservations'));
    }
}
