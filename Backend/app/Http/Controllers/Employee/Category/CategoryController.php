<?php

namespace App\Http\Controllers\Employee\Category;

use App\Http\Controllers\Controller;
use App\Models\Category;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class CategoryController extends Controller
{
    //

    public function index()
    {
        $categories = Category::all();
        return view('Employee.Category.index', compact('categories'));
    }


    public function create()
    {
        return view('Employee.Category.create');
    }

    public function store(Request $request)
    {
        Category::create([
            'name' => $request->input('name'),
            'type' => $request->input('type'),
            'created_By' => Auth::guard('employee')->user()->id,
        ]);

        return redirect()->route('employee.food.index')->with('success_message', 'Category Created Successfully');
    }



    public function edit($id)
    {
        $category = Category::findOrfail($id);
        return view('employee.category.edit', compact('category'));
    }



    public function update(Request $request, $id)
    {

            $category = Category::findOrFail($id);

            $category->update([
                'name' => $request->input('name'),
                'type' => $request->input('type'),
            ]);

            return redirect()->route('employee.category.index')->with('success_message', 'Category Updated Successfully');
    
        
    }



    public function archive()
    {
        $categories = Category::onlyTrashed()->get();
        return view('Employee.Category.archive', compact('categories'));
    }



    public function delete($id)
    {
        $category = Category::findOrFail($id);

        $category->delete();

        return redirect()->route('employee.category.index')->with('success_message', 'Category Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $category = Category::withTrashed()->where('id', $id)->first();
        if ($category) {
            $category->forceDelete();

            return redirect()->route('employee.category.archive')->with('success_message', 'Category Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Category Not Found');
        }
    }



    public function restore($id)
    {
        Category::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.category.archive')->with('success_message', 'Category Restored Successfully');
    }
}
