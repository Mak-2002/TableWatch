<?php

namespace App\Http\Controllers\Employee\User;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Models\UserR;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Storage;

class UserController extends Controller
{
    //

    public function index()
    {
        $users = UserR::all();
        return view('Employee.User.index', compact('users'));
    }


    public function create()
    {
        return view('Employee.User.create');
    }

    public function store(Request $request)
    {
        $password = $request->input('password');
        $image = $request->file('img')->getClientOriginalName();
        $path = $request->file('img')->storeAs('User', $image, 'image');

        UserR::create([
            'name' => $request->input('name'),
            'email' => $request->input('email'),
            'password' => Hash::make($password),
            'phone' => $request->input('phone'),
            'age' => $request->input('age'),
            'gender' => $request->input('gender'),
            'status' => $request->input('status'),
            'img' => $path,
            'created_by' => Auth::guard('employee')->user()->id,
        ]);

        return redirect()->route('employee.user.index')->with('success_message', 'User Created Successfully');
    }



    public function edit($id)
    {
        $user = UserR::findOrfail($id);
        return view('employee.user.edit', compact('user'));
    }



    public function update(Request $request, $id)
    {

        $user = UserR::findOrFail($id);

        if ($request->file('img') == null) {
            $user->update([
                'name' => $request->input('name'),
                'email' => $request->input('email'),
                'phone' => $request->input('phone'),
                'age' => $request->input('age'),
                'gender' => $request->input('gender'),
                'status' => $request->input('status'),
            ]);

            return redirect()->route('employee.user.index')->with('success_message', 'User Updated Successfully');
        } else {
            if ($user->img != null) {
                Storage::disk('image')->delete($user->img);
                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('User', $image, 'image');

                $user->update([
                    'name' => $request->input('name'),
                    'email' => $request->input('email'),
                    'phone' => $request->input('phone'),
                    'age' => $request->input('age'),
                    'gender' => $request->input('gender'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('employee.user.index')->with('success_message', 'User Updated Successfully');
            } else {

                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('User', $image, 'image');

                $user->update([
                    'name' => $request->input('name'),
                    'email' => $request->input('email'),
                    'phone' => $request->input('phone'),
                    'age' => $request->input('age'),
                    'gender' => $request->input('gender'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('employee.user.index')->with('success_message', 'User Updated Successfully');
            }
        }
    }



    public function archive()
    {
        $users = UserR::onlyTrashed()->get();
        return view('Employee.User.archive', compact('users'));
    }



    public function delete($id)
    {
        $user = UserR::findOrFail($id);

        $user->delete();

        return redirect()->route('employee.user.index')->with('success_message', 'User Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $user = UserR::withTrashed()->where('id', $id)->first();
        if ($user) {
            Storage::disk('image')->delete($user->img);
            $user->forceDelete();

            return redirect()->route('employee.user.archive')->with('success_message', 'User Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'User Not Found');
        }
    }



    public function restore($id)
    {
        UserR::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.user.archive')->with('success_message', 'User Restored Successfully');
    }
}
