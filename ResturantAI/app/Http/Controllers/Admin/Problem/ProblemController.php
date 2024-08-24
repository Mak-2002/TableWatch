<?php

namespace App\Http\Controllers\Admin\Problem;

use App\Http\Controllers\Controller;
use App\Models\Problem;
use Illuminate\Http\Request;

class ProblemController extends Controller
{
    //
    public function index()
    {
        $problems = Problem::all();
        return view('Admin.Problem.index' , compact('problems'));
    }
}
