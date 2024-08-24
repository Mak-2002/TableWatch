<?php

namespace App\Http\Controllers\Employee\Reservations;

use App\Http\Controllers\Controller;
use App\Models\Reservation;
use App\Models\Table;
use App\Models\UserR;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class ReservationController extends Controller
{
    //
    public function index()
    {
        $reservations = Reservation::all();
        return view('Employee.Reservation.index', compact('reservations'));
    }


    public function create()
    {
        $tables = Table::all();
        $users = UserR::all();
        return view('Employee.Reservation.create', compact('tables', 'users'));
    }

    public function store(Request $request)
    {
        $day = $request->input('day');
        $date = $request->input('date');
        $timeStart = $request->input('timeStart');
        $timeEnd = $request->input('timeEnd');
        $tableID = $request->input('tableID');
        $userID = $request->input('userID');


        $ifExists = Reservation::where('date', $date)
            ->where('tableID', $tableID)
            ->where(function ($query) use ($timeStart, $timeEnd) {
                $query->whereBetween('timeStart', [$timeStart, $timeEnd])
                    ->orWhereBetween('timeEnd', [$timeStart, $timeEnd])
                    ->orWhere(function ($query) use ($timeStart, $timeEnd) {
                        $query->where('timeStart', '<=', $timeStart)
                            ->where('timeEnd', '>=', $timeEnd);
                    });
            })
            ->exists();

        if ($ifExists) {
            return redirect()->route('employee.reservation.index')->with('error_message', 'This table is already reserved at the requested time.');
        }


        Reservation::create([
            'day' => $day,
            'date' => $date,
            'timeStart' => $timeStart,
            'timeEnd' => $timeEnd,
            'status' => '0',
            'created_By' => Auth::guard('employee')->user()->id,
            'tableID' => $tableID,
            'userID' => $userID,
        ]);

        return redirect()->route('employee.reservation.index')->with('success_message', 'Reservation Created Successfully');
    }




    public function edit($id)
    {
        $reservation = Reservation::findOrfail($id);
        return view('employee.reservation.edit', compact('reservation'));
    }



    public function update(Request $request, $id)
    {
        $reservation = Reservation::findOrFail($id);

        $day = $request->input('day');
        $date = $request->input('date');
        $timeStart = $request->input('timeStart');
        $timeEnd = $request->input('timeEnd');
        $tableID = $request->input('tableID');
        $userID = $request->input('userID');

        $ifExists = Reservation::where('date', $date)
            ->where('tableID', $tableID)
            ->where('id', '!=', $id)
            ->where(function ($query) use ($timeStart, $timeEnd) {
                $query->whereBetween('timeStart', [$timeStart, $timeEnd])
                    ->orWhereBetween('timeEnd', [$timeStart, $timeEnd])
                    ->orWhere(function ($query) use ($timeStart, $timeEnd) {
                        $query->where('timeStart', '<=', $timeStart)
                            ->where('timeEnd', '>=', $timeEnd);
                    });
            })
            ->exists();

        if ($ifExists) {
            return redirect()->route('employee.reservation.index')->with('error_message', 'This table is already reserved at the requested time.');
        }


        $reservation->update([
            'day' => $day,
            'date' => $date,
            'timeStart' => $timeStart,
            'timeEnd' => $timeEnd,
            'status' => $request->input('status'),
            'tableID' => $tableID,
            'userID' => $userID,
        ]);

        return redirect()->route('employee.reservation.index')->with('success_message', 'Reservation Updated Successfully');
    }




    public function archive()
    {
        $reservations = Reservation::onlyTrashed()->get();
        return view('Employee.reservation.archive', compact('reservations'));
    }



    public function delete($id)
    {
        $reservation = Reservation::findOrFail($id);

        $reservation->delete();

        return redirect()->route('employee.reservation.index')->with('success_message', 'Reservation Deleted Successfully');
    }

    public function cancel($id)
    {
        $reservation = Reservation::findOrFail($id);

        $reservation->update([
            'status' => '2'
        ]);

        return redirect()->route('employee.reservation.index')->with('success_message', 'Reservation Canceled Successfully');
    }



    public function forceDelete($id)
    {
        $reservation = Reservation::withTrashed()->where('id', $id)->first();
        if ($reservation) {
            $reservation->forceDelete();

            return redirect()->route('employee.reservation.archive')->with('success_message', 'Reservation Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Reservation Not Found');
        }
    }



    public function restore($id)
    {
        Reservation::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.reservation.archive')->with('success_message', 'Reservation Restored Successfully');
    }
}
