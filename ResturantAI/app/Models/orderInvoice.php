<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class orderInvoice extends Model
{
    use HasFactory, SoftDeletes;

    protected $fillable = [
        'typePayment',
        'tax',
        'totalAmmount',
        'status',
        'waiterID',
        'reservationID',
        'created_By',
    ];

    public function employee()
    {
        return $this->belongsTo(Employee::class, 'created_By');
    }

    public function reservation()
    {
        return $this->belongsTo(Reservation::class, 'reservationID');
    }

    public function waiter()
    {
        return $this->belongsTo(Waiter::class, 'waiterID');
    }

    public function orderInvoiceDetails()
    {
        return $this->hasMany(orderInvoiceDetails::class, 'orderInvoiceID');
    }
}
