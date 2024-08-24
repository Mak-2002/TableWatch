<?php

namespace App\Http\Controllers\Employee\Auth;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class AuthController extends Controller
{
    //


      public function login_page()
      {
          return view('Employee.Auth.login');
      }
  
  
      public function login_check(Request $request)
      {
          $check = $request->all();
 
          if (Auth::guard('employee')->attempt(['email' => $check['email'], 'password' => $check['password']])) {
              return redirect()->route('employee.dashboard');
          } else {
              return redirect()->route('employee.login.page')->with('error_message', 'Check Email Or Password');
          }
      }

  
      public function logout()
      {
          Auth::guard('employee')->logout();
          return redirect()->route('employee.login.page');
      }
}
