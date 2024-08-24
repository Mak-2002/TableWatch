<?php

namespace App\Http\Controllers\Employee\OrderInvoice;

use App\Http\Controllers\Controller;
use App\Models\FoodCategory;
use App\Models\orderInvoice;
use App\Models\orderInvoiceDetails;
use App\Models\Reservation;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class OrderInvoiceController extends Controller
{
    //

    public function index()
    {
        $orderInvoices = orderInvoice::all();
        return view('Employee.OrderInvoice.index', compact('orderInvoices'));
    }

    public function create()
    {
        $reservations = Reservation::where('status', 0)->get();
        return view('Employee.OrderInvoice.create', compact('reservations'));
    }

    public function store(Request $request)
    {
        orderInvoice::create([
            'reservationID' => $request->input('reservationID'),
            'status' => '0',
            'created_By' => Auth::guard('employee')->user()->id,
        ]);
        return redirect()->route('employee.order.invoice.index')->with('success_message', 'Order Invoice Created Successfully');
    }


    public function foodDetails($id)
    {
        $orderInvoiceID = $id;
        $orderInvoiceDetails = orderInvoiceDetails::where('orderInvoiceID', $orderInvoiceID)->get();
        return view('Employee.OrderInvoice.OrderInvoiceDetails.index', compact('orderInvoiceDetails', 'orderInvoiceID'));
    }

    public function addFood($id)
    {
        $orderInvoiceID = $id;
        $foodCategories = FoodCategory::all();
        return view('Employee.OrderInvoice.OrderInvoiceDetails.createFood', compact('orderInvoiceID', 'foodCategories'));
    }

    public function storeFood(Request $request)
    {
        $orderInvoiceID = $request->input('orderInvoiceID');
        $foodQuantity = $request->input('foodQuantity');
        $foodNote = $request->input('foodNote');
        $foodAmmount = $request->input('foodAmmount');
        $foodCategoryID = $request->input('foodCategoryID');

        orderInvoiceDetails::create([
            'foodQuantity' => $foodQuantity,
            'foodNote' => $foodNote,
            'foodAmmount' => $foodAmmount,
            'orderInvoiceID' => $orderInvoiceID,
            'foodCategoryID' => $foodCategoryID,
        ]);

        return redirect()->route('employee.order.invoice.index')->with('success_message', 'Order Invoice Food Created Successfully');
    }

    public function deleteFood($id)
    {
        $orderInvoiceDetails = orderInvoiceDetails::findOrFail($id);

        $orderInvoiceDetails->delete();

        return redirect()->back()->with('success_message', 'Order Invoice Details Deleted Successfully');
    }

    public function finish ($id)
    {
        $orderInvoiceID = $id;
        $orderInvoice = orderInvoice::findOrFail($orderInvoiceID);
        $orderInvoice->update([
            'status' => '1'
        ]);

        return redirect()->back()->with('success_message', 'Order Invoice Finished Successfully');
    }

    public function archive()
    {
        $orderInvoices = orderInvoice::onlyTrashed()->get();
        return view('Employee.OrderInvoice.archive', compact('orderInvoices'));
    }


    public function delete($id)
    {
        $orderInvoice = orderInvoice::findOrFail($id);

        $orderInvoice->delete();

        return redirect()->route('employee.order.invoice.index')->with('success_message', 'Order Invoice Deleted Successfully');
    }


    public function forceDelete($id)
    {
        $orderInvoice = orderInvoice::withTrashed()->where('id', $id)->first();
        if ($orderInvoice) {
            $orderInvoice->forceDelete();

            return redirect()->route('employee.order.invoice.archive')->with('success_message', 'Order Invoice Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Order Invoice Not Found');
        }
    }


    public function restore($id)
    {
        orderInvoice::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.order.invoice.archive')->with('success_message', 'Order Invoice Restored Successfully');
    }
}
