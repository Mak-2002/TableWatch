<?php

namespace App\Http\Controllers\Employee\Table;

use App\Http\Controllers\Controller;
use App\Models\Table;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class TableController extends Controller
{
    //

    public function index()
    {
        $tables = Table::all();
        return view('Employee.Table.index', compact('tables'));
    }


    public function create()
    {
        return view('Employee.Table.create');
    }

    public function store(Request $request)
    {
        $name = $request->input('name');
        $number = $request->input('number');
        $capacity = $request->input('capacity');

        $checkExists = Table::where('name', $name)->orwhere('number', $number)->first();

        if ($checkExists) {
            return redirect()->route('employee.table.index')->with('error_message', 'Table Already Exists');
        }

        Table::create([
            'name' => $name,
            'number' => $number,
            'capacity' => $capacity,
            'created_By' => Auth::guard('employee')->user()->id,
        ]);

        return redirect()->route('employee.table.index')->with('success_message', 'Table Created Successfully');
    }


    public function edit($id)
    {
        $table = Table::findOrfail($id);
        return view('employee.table.edit', compact('table'));
    }

    public function update(Request $request, $id)
    {

        $table = Table::findOrFail($id);

        $name = $request->input('name');
        $number = $request->input('number');
        $capacity = $request->input('capacity');

        $checkExists = Table::where('name', $name)->orwhere('number', $number)->first();

        if ($checkExists) {
            return redirect()->route('employee.table.index')->with('error_message', 'Table Already Exists');
        }

        $table->update([
            'name' => $name,
            'number' => $number,
            'capacity' => $capacity,
        ]);

        return redirect()->route('employee.table.index')->with('success_message', 'Table Updated Successfully');
    }



    public function archive()
    {
        $tables = Table::onlyTrashed()->get();
        return view('Employee.table.archive', compact('tables'));
    }



    public function delete($id)
    {
        $table = Table::findOrFail($id);

        $table->delete();

        return redirect()->route('employee.table.index')->with('success_message', 'Table Deleted Successfully');
    }

    public function forceDelete($id)
    {
        $table = Table::withTrashed()->where('id', $id)->first();
        if ($table) {
            $table->forceDelete();

            return redirect()->route('employee.table.archive')->with('success_message', 'Table Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Table Not Found');
        }
    }



    public function restore($id)
    {
        Table::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.table.archive')->with('success_message', 'Table Restored Successfully');
    }
}
