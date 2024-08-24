<?php

namespace App\Http\Controllers\Employee\FoodCategory;

use App\Http\Controllers\Controller;
use App\Models\Category;
use App\Models\Food;
use App\Models\FoodCategory;
use Illuminate\Http\Request;

class FoodCategoryController extends Controller
{
    //
    public function index()
    {
        $categoryFoods = FoodCategory::all();
        return view('Employee.FoodCategory.index', compact('categoryFoods'));
    }


    public function create()
    {
        $foods = Food::all();
        $categories = Category::all();
        return view('Employee.FoodCategory.create', compact('foods', 'categories'));
    }

    public function store(Request $request)
    {
        $foodID = $request->input('foodID');
        $categoryID = $request->input('categoryID');
        $checkExists = FoodCategory::where('foodID', $foodID)->where('categoryID', $categoryID)->first();
        if ($checkExists) {
            return redirect()->back()->with('error_message', 'Food Already Have This Category');
        }
        FoodCategory::create([
            'note' => $request->input('note'),
            'price' => $request->input('price'),
            'foodID' => $foodID,
            'categoryID' => $categoryID,
        ]);

        return redirect()->route('employee.food.category.index')->with('success_message', 'Food Category Created Successfully');
    }


    public function archive()
    {
        $categoryFoods = FoodCategory::onlyTrashed()->get();
        return view('Employee.FoodCategory.archive', compact('categoryFoods'));
    }



    public function delete($id)
    {
        $categoryFood = FoodCategory::findOrFail($id);

        $categoryFood->delete();

        return redirect()->route('employee.food.category.index')->with('success_message', 'Food Category Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $categoryFood = FoodCategory::withTrashed()->where('id', $id)->first();
        if ($categoryFood) {
            $categoryFood->forceDelete();

            return redirect()->route('employee.food.category.archive')->with('success_message', 'Food Category Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Food Category Not Found');
        }
    }



    public function restore($id)
    {
        FoodCategory::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.food.category.archive')->with('success_message', 'Food Category Restored Successfully');
    }
}
