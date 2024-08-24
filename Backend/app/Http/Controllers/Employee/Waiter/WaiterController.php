<?php

namespace App\Http\Controllers\Employee\Waiter;

use App\Http\Controllers\Controller;
use App\Models\Waiter;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;
use Intervention\Image\ImageManagerStatic as Image;


class WaiterController extends Controller
{
    public function index()
    {
        $waiters = Waiter::all();
        return view('Employee.Waiter.index', compact('waiters'));
    }


    public function create()
    {
        return view('Employee.Waiter.create');
    }

    public function store(Request $request)
{
    $validatedData = $request->validate([
        'name' => 'required|string|max:255',
        'email' => 'required|email|unique:waiters,email',
        'phone' => 'required|string|max:20',
        'age' => 'required|integer|min:18',
        'gender' => 'required|boolean',
        'status' => 'required|boolean',
        'faceImage' => 'required|string', // Assuming the faceImage is a base64 encoded string
    ]);

    // Process and save face image
    if ($request->has('faceImage')) {
        $faceImageData = $validatedData['faceImage'];
        $faceImage = base64_decode(preg_replace('#^data:image/\w+;base64,#i', '', $faceImageData));
        $waiterId = Waiter::max('id') + 1; // Get the next ID
        $faceImageName = $waiterId . '.jpg';
        $facePath = 'waiterPhoto/' . $faceImageName;
        Storage::disk('image')->put($facePath, $faceImage);

        $waiter = Waiter::create([
            'name' => $validatedData['name'],
            'email' => $validatedData['email'],
            'phone' => $validatedData['phone'],
            'age' => $validatedData['age'],
            'gender' => $validatedData['gender'],
            'status' => $validatedData['status'],
            'facePath' => $facePath,
            'created_By' => Auth::guard('employee')->user()->id,
        ]);

        return redirect()->route('employee.waiter.index')->with('success_message', 'Waiter Created Successfully');
    }

    return back()->withErrors('Image upload failed');
}



    public function archive()
    {
        $waiters = Waiter::onlyTrashed()->get();
        return view('Employee.Waiter.archive', compact('waiters'));
    }



    public function delete($id)
    {
        $waiter = Waiter::findOrFail($id);

        $waiter->delete();

        return redirect()->route('employee.waiter.index')->with('success_message', 'Waiter Deleted Successfully');
    }



    public function forceDelete($id)
    {
        $waiter = Waiter::withTrashed()->where('id', $id)->first();
        if ($waiter) {
            Storage::disk('image')->delete($waiter->img);
            $waiter->forceDelete();

            return redirect()->route('employee.waiter.archive')->with('success_message', 'Waiter Deleted Successfully');
        } else {
            return redirect()->back()->with('error_message', 'Waiter Not Found');
        }
    }


    public function restore($id)
    {
        Waiter::withTrashed()->where('id', $id)->restore();

        return redirect()->route('employee.waiter.archive')->with('success_message', 'Waiter Restored Successfully');
    }

//     public function uploadPhoto(Request $request)
// {
//     $request->validate([
//         'photo' => 'required|image|mimes:jpg,png,jpeg|max:2048',
//     ]);

//     $photo = $request->file('photo');
//     $photoName = uniqid() . '.' . $photo->getClientOriginalExtension();
//     $path = $photo->storeAs('waiterPhoto', $photoName, 'public');

//     return response()->json(['path' => $path]);
// }

}
