<?php

namespace App\Http\Controllers\Admin\Employee;

use App\Http\Controllers\Controller;
use App\Models\Employee;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Storage;

class EmployeeController extends Controller
{
    //

    public function index()
    {
        $employees = Employee::all();
        return view('Admin.Employee.index', compact('employees'));
    }


    public function create()
    {
        return view('Admin.Employee.create');
    }

    public function store(Request $request)
    {
        $password = $request->input('password');
        $image = $request->file('img')->getClientOriginalName();
        $path = $request->file('img')->storeAs('Employee', $image, 'image');

        Employee::create([
            'name' => $request->input('name'),
            'email' => $request->input('email'),
            'password' => Hash::make($password),
            'phone' => $request->input('phone'),
            'age' => $request->input('age'),
            'gender' => $request->input('gender'),
            'status' => $request->input('status'),
            'img' => $path,
            'created_By' => Auth::guard('admin')->user()->id,
        ]);

        return redirect()->route('admin.employee.index')->with('success_message', 'Employee Created Successfully');
    }



    public function edit($id)
    {
        $employee = Employee::findOrfail($id);
        return view('admin.employee.edit', compact('employee'));
    }



    public function update(Request $request, $id)
    {

        $employee = Employee::findOrFail($id);

        if ($request->file('img') == null) {
            $employee->update([
                'name' => $request->input('name'),
                'email' => $request->input('email'),
                'phone' => $request->input('phone'),
                'age' => $request->input('age'),
                'gender' => $request->input('gender'),
                'status' => $request->input('status'),
            ]);

            return redirect()->route('admin.employee.index')->with('success_message', 'Employee Updated Successfully');
        } else {
            if ($employee->img != null) {
                Storage::disk('image')->delete($employee->img);
                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('Employee', $image, 'image');

                $employee->update([
                    'name' => $request->input('name'),
                    'email' => $request->input('email'),
                    'phone' => $request->input('phone'),
                    'age' => $request->input('age'),
                    'gender' => $request->input('gender'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('admin.employee.index')->with('success_message', 'Employee Updated Successfully');
            } else {

                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('Employee', $image, 'image');

                $employee->update([
                    'name' => $request->input('name'),
                    'email' => $request->input('email'),
                    'phone' => $request->input('phone'),
                    'age' => $request->input('age'),
                    'gender' => $request->input('gender'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('admin.employee.index')->with('success_message', 'Employee Updated Successfully');
            }
        }
    }



    public function archive()
    {
        $employees = Employee::onlyTrashed()->get();
        return view('Admin.Employee.archive', compact('employees'));
    }



    public function delete($id)
    {
        $employee = Employee::findOrFail($id);

        $employee->delete();

        return redirect()->route('admin.employee.index')->with('success_message', 'Employee Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $employee = Employee::withTrashed()->where('id', $id)->first();
        if ($employee) {
            Storage::disk('image')->delete($employee->img);
            $employee->forceDelete();

            return redirect()->route('admin.employee.archive')->with('success_message', 'Employee Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Employe Not Found');
        }
    }



    public function restore($id)
    {
        Employee::withTrashed()->where('id', $id)->restore();

        return redirect()->route('admin.employee.archive')->with('success_message', 'Employee Restored Successfully');
    }
}
