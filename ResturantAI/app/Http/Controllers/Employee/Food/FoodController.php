<?php

namespace App\Http\Controllers\Employee\Food;

use App\Http\Controllers\Controller;
use App\Models\Food;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;

class FoodController extends Controller
{
    //


    public function index()
    {
        $foods = Food::all();
        return view('Employee.Food.index', compact('foods'));
    }


    public function create()
    {
        return view('Employee.Food.create');
    }

    public function store(Request $request)
    {
        $image = $request->file('img')->getClientOriginalName();
        $path = $request->file('img')->storeAs('Food', $image, 'image');

        Food::create([
            'name' => $request->input('name'),
            'details' => $request->input('details'),
            'note' => $request->input('note'),
            'status' => $request->input('status'),
            'img' => $path,
            'created_By' => Auth::guard('employee')->user()->id,
        ]);

        return redirect()->route('employee.food.index')->with('success_message', 'Food Created Successfully');
    }



    public function edit($id)
    {
        $food = Food::findOrfail($id);
        return view('employee.food.edit', compact('food'));
    }



    public function update(Request $request, $id)
    {

        $food = Food::findOrFail($id);

        if ($request->file('img') == null) {
            $food->update([
                'name' => $request->input('name'),
                'details' => $request->input('details'),
                'note' => $request->input('note'),
                'status' => $request->input('status'),
            ]);

            return redirect()->route('employee.food.index')->with('success_message', 'Food Updated Successfully');
        } else {
            if ($food->img != null) {
                Storage::disk('image')->delete($food->img);
                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('Food', $image, 'image');

                $food->update([
                    'name' => $request->input('name'),
                    'details' => $request->input('details'),
                    'note' => $request->input('note'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('employee.food.index')->with('success_message', 'Food Updated Successfully');
            } else {

                $image = $request->file('img')->getClientOriginalName();
                $path = $request->file('img')->storeAs('Food', $image, 'image');

                $food->update([
                    'name' => $request->input('name'),
                    'details' => $request->input('details'),
                    'note' => $request->input('note'),
                    'status' => $request->input('status'),
                    'img' => $path,
                ]);

                return redirect()->route('employee.food.index')->with('success_message', 'Food Updated Successfully');
            }
        }
    }



    public function archive()
    {
        $foods = Food::onlyTrashed()->get();
        return view('Employee.Food.archive', compact('foods'));
    }



    public function delete($id)
    {
        $food = Food::findOrFail($id);

        $food->delete();

        return redirect()->route('employee.food.index')->with('success_message', 'Food Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $food = Food::withTrashed()->where('id', $id)->first();
        if ($food) {
            Storage::disk('image')->delete($food->img);
            $food->forceDelete();

            return redirect()->route('employee.food.archive')->with('success_message', 'Food Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Food Not Found');
        }
    }



    public function restore($id)
    {
        Food::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.food.archive')->with('success_message', 'Food Restored Successfully');
    }
}
