<?php

use App\Http\Controllers\Employee\Auth\AuthController;
use App\Http\Controllers\Employee\Category\CategoryController;
use App\Http\Controllers\Employee\EmployeeController;
use App\Http\Controllers\Employee\Food\FoodController;
use App\Http\Controllers\Employee\FoodCategory\FoodCategoryController;
use App\Http\Controllers\Employee\OrderInvoice\OrderInvoiceController;
use App\Http\Controllers\Employee\Reservations\ReservationController;
use App\Http\Controllers\Employee\Table\TableController;
use App\Http\Controllers\Employee\User\UserController;
use App\Http\Controllers\Employee\Waiter\WaiterController;
use App\Models\FoodCategory;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Employee Routes
|--------------------------------------------------------------------------
|
| Here is where you can register Employee routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "Employee" middleware group. Make something great!
|
*/


//=================== Auth Route ==============
Route::get('employee/login', [AuthController::class, 'login_page'])->name('employee.login.page');
Route::post('employee/login/check', [AuthController::class, 'login_check'])->name('employee.login.check');



Route::middleware(['Employee'])->name('employee.')->prefix('employee')->group(function () {

    Route::get('/dashboard', [EmployeeController::class, 'index'])->name('dashboard');
    Route::get('/logout', [AuthController::class, 'logout'])->name('logout');

    //============================ Reservations Route ========================

    Route::get('/reservation/index', [ReservationController::class, 'index'])->name('reservation.index');

    Route::get('/reservation/create', [ReservationController::class, 'create'])->name('reservation.create');

    Route::post('/reservation/store', [ReservationController::class, 'store'])->name('reservation.store');

    Route::get('/reservation/edit/{id}', [ReservationController::class, 'edit'])->name('reservation.edit');

    Route::put('/reservation/update/{id}', [ReservationController::class, 'update'])->name('reservation.update');

    Route::delete('/reservation/delete/{id}', [ReservationController::class, 'delete'])->name('reservation.delete');

    Route::put('/reservation/cancel/{id}', [ReservationController::class, 'cancel'])->name('reservation.cancel');

    Route::get('/reservation/archive', [ReservationController::class, 'archive'])->name('reservation.archive');

    Route::get('/reservation/restore/{id}', [ReservationController::class, 'restore'])->name('reservation.restore');

    Route::delete('/reservation/force/delete/{id}', [ReservationController::class, 'forceDelete'])->name('reservation.forceDelete');

    //============================ Table Route ========================

    Route::get('/table/index', [TableController::class, 'index'])->name('table.index');

    Route::get('/table/create', [TableController::class, 'create'])->name('table.create');

    Route::post('/table/store', [TableController::class, 'store'])->name('table.store');

    Route::get('/table/edit/{id}', [TableController::class, 'edit'])->name('table.edit');

    Route::put('/table/update/{id}', [TableController::class, 'update'])->name('table.update');

    Route::delete('/table/delete/{id}', [TableController::class, 'delete'])->name('table.delete');

    Route::get('/table/archive', [TableController::class, 'archive'])->name('table.archive');

    Route::get('/table/restore/{id}', [TableController::class, 'restore'])->name('table.restore');

    Route::delete('/table/force/delete/{id}', [TableController::class, 'forceDelete'])->name('table.forceDelete');


    //============================ User Route ========================

    Route::get('/user/index', [UserController::class, 'index'])->name('user.index');

    Route::get('/user/create', [UserController::class, 'create'])->name('user.create');

    Route::post('/user/store', [UserController::class, 'store'])->name('user.store');

    Route::get('/user/edit/{id}', [UserController::class, 'edit'])->name('user.edit');

    Route::put('/user/update/{id}', [UserController::class, 'update'])->name('user.update');

    Route::delete('/user/delete/{id}', [UserController::class, 'delete'])->name('user.delete');

    Route::get('/user/archive', [UserController::class, 'archive'])->name('user.archive');

    Route::get('/user/restore/{id}', [UserController::class, 'restore'])->name('user.restore');

    Route::delete('/user/force/delete/{id}', [UserController::class, 'forceDelete'])->name('user.forceDelete');


    //============================ Waiter Route ========================

    Route::get('/waiter/index', [WaiterController::class, 'index'])->name('waiter.index');

    Route::get('/waiter/create', [WaiterController::class, 'create'])->name('waiter.create');

    Route::post('/waiter/store', [WaiterController::class, 'store'])->name('waiter.store');

    Route::get('/waiter/edit/{id}', [WaiterController::class, 'edit'])->name('waiter.edit');

    Route::put('/waiter/update/{id}', [WaiterController::class, 'update'])->name('waiter.update');

    Route::delete('/waiter/delete/{id}', [WaiterController::class, 'delete'])->name('waiter.delete');

    Route::get('/waiter/archive', [WaiterController::class, 'archive'])->name('waiter.archive');

    Route::get('/waiter/restore/{id}', [WaiterController::class, 'restore'])->name('waiter.restore');

    Route::delete('/waiter/force/delete/{id}', [WaiterController::class, 'forceDelete'])->name('waiter.forceDelete');


    //============================ Food Route ========================

    Route::get('/food/index', [FoodController::class, 'index'])->name('food.index');

    Route::get('/food/create', [FoodController::class, 'create'])->name('food.create');

    Route::post('/food/store', [FoodController::class, 'store'])->name('food.store');

    Route::get('/food/edit/{id}', [FoodController::class, 'edit'])->name('food.edit');

    Route::put('/food/update/{id}', [FoodController::class, 'update'])->name('food.update');

    Route::delete('/food/delete/{id}', [FoodController::class, 'delete'])->name('food.delete');

    Route::get('/food/archive', [FoodController::class, 'archive'])->name('food.archive');

    Route::get('/food/restore/{id}', [FoodController::class, 'restore'])->name('food.restore');

    Route::delete('/food/force/delete/{id}', [FoodController::class, 'forceDelete'])->name('food.forceDelete');


    //============================ Category Route ========================

    Route::get('/category/index', [CategoryController::class, 'index'])->name('category.index');

    Route::get('/category/create', [CategoryController::class, 'create'])->name('category.create');

    Route::post('/category/store', [CategoryController::class, 'store'])->name('category.store');

    Route::get('/category/edit/{id}', [CategoryController::class, 'edit'])->name('category.edit');

    Route::put('/category/update/{id}', [CategoryController::class, 'update'])->name('category.update');

    Route::delete('/category/delete/{id}', [CategoryController::class, 'delete'])->name('category.delete');

    Route::get('/category/archive', [CategoryController::class, 'archive'])->name('category.archive');

    Route::get('/category/restore/{id}', [CategoryController::class, 'restore'])->name('category.restore');

    Route::delete('/category/force/delete/{id}', [CategoryController::class, 'forceDelete'])->name('category.forceDelete');


    //============================ Food Category Route ========================

    Route::get('/food/category/index', [FoodCategoryController::class, 'index'])->name('food.category.index');

    Route::get('/food/category/create', [FoodCategoryController::class, 'create'])->name('food.category.create');

    Route::post('/food/category/store', [FoodCategoryController::class, 'store'])->name('food.category.store');

    Route::delete('/food/category/delete/{id}', [FoodCategoryController::class, 'delete'])->name('food.category.delete');

    Route::get('/food/category/archive', [FoodCategoryController::class, 'archive'])->name('food.category.archive');

    Route::get('/food/category/restore/{id}', [FoodCategoryController::class, 'restore'])->name('food.category.restore');

    Route::delete('/food/category/force/delete/{id}', [FoodCategoryController::class, 'forceDelete'])->name('food.category.forceDelete');


    //============================ Order Route ========================

    Route::get('/order/invoice/index', [OrderInvoiceController::class, 'index'])->name('order.invoice.index');

    Route::get('/order/invoice/create', [OrderInvoiceController::class, 'create'])->name('order.invoice.create');

    Route::post('/order/invoice/store', [OrderInvoiceController::class, 'store'])->name('order.invoice.store');

    Route::get('/order/invoice/foodDetails/{id}', [OrderInvoiceController::class, 'foodDetails'])->name('order.invoice.food.details');

    Route::get('/order/invoice/add/food/{id}', [OrderInvoiceController::class, 'addFood'])->name('order.invoice.add.food');

    Route::post('/order/invoice/store/food', [OrderInvoiceController::class, 'storeFood'])->name('order.invoice.store.food');

    Route::delete('/order/invoice/delete/food/{id}', [OrderInvoiceController::class, 'deleteFood'])->name('order.invoice.delete.food');

    Route::put('/order/invoice/finish/{id}', [OrderInvoiceController::class, 'finish'])->name('order.invoice.finish');

    Route::delete('/order/invoice/delete/{id}', [OrderInvoiceController::class, 'delete'])->name('order.invoice.delete');

    Route::get('/order/invoice/archive', [OrderInvoiceController::class, 'archive'])->name('order.invoice.archive');

    Route::get('/order/invoice/restore/{id}', [OrderInvoiceController::class, 'restore'])->name('order.invoice.restore');

    Route::delete('/order/invoice/force/delete/{id}', [OrderInvoiceController::class, 'forceDelete'])->name('order.invoice.forceDelete');
});
