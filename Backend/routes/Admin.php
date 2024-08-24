<?php

use App\Http\Controllers\Admin\AdminController;
use App\Http\Controllers\Admin\Auth\AuthController;
use App\Http\Controllers\Admin\Employee\EmployeeController;
use App\Http\Controllers\Admin\Problem\ProblemController;
use App\Http\Controllers\Admin\Reservation\ReservationController;
use App\Models\Problem;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Admin Routes
|--------------------------------------------------------------------------
|
| Here is where you can register Admin routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "Admin" middleware group. Make something great!
|
*/


//=================== Auth Route ==============
Route::get('admin/login', [AuthController::class, 'login_page'])->name('admin.login.page');
Route::post('admin/login/check', [AuthController::class, 'login_check'])->name('admin.login.check');



Route::middleware(['Admin'])->name('admin.')->prefix('admin')->group(function () {

  Route::get('/dashboard', [AdminController::class, 'index'])->name('dashboard');
  Route::get('/logout', [AuthController::class, 'logout'])->name('logout');


  //============================ Employee Route ========================

  Route::get('/employee/index', [EmployeeController::class, 'index'])->name('employee.index');

  Route::get('/employee/create', [EmployeeController::class, 'create'])->name('employee.create');

  Route::post('/employee/store', [EmployeeController::class, 'store'])->name('employee.store');

  Route::get('/employee/edit/{id}', [EmployeeController::class, 'edit'])->name('employee.edit');

  Route::put('/employee/update/{id}', [EmployeeController::class, 'update'])->name('employee.update');

  Route::delete('/employee/delete/{id}', [EmployeeController::class, 'delete'])->name('employee.delete');

  Route::get('/employee/archive', [EmployeeController::class, 'archive'])->name('employee.archive');

  Route::get('/employee/restore/{id}', [EmployeeController::class, 'restore'])->name('employee.restore');

  Route::delete('/employee/force/delete/{id}', [EmployeeController::class, 'forceDelete'])->name('employee.force.delete');


  //============================ Reservations Route ========================

  Route::get('/reservation/index', [ReservationController::class, 'index'])->name('reservation.index');


  //============================ Show Problem Route ========================

  Route::get('/problem/index', [ProblemController::class, 'index'])->name('problem.index');

});
