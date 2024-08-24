<?php

namespace App\Http\Controllers\Admin\Auth;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class AuthController extends Controller
{
    //

     //عرض صفحة تسجيل الدخول 

     public function login_page()
     {
         return view('Admin.Auth.login');
     }
 
     //التحقق من عملية تسجيل الدخول
 
     public function login_check(Request $request)
     {
         $check = $request->all();

         if (Auth::guard('admin')->attempt(['email' => $check['email'], 'password' => $check['password']])) {
             return redirect()->route('admin.dashboard');
         } else {
             return redirect()->route('admin.login.page')->with('error_message', 'Check Email Or Password');
         }
     }
 
     //تسجيل الخروج
 
     public function logout()
     {
         Auth::guard('admin')->logout();
         return redirect()->route('admin.login.page');
     }

}
