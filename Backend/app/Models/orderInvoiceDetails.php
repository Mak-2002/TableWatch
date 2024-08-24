<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class orderInvoiceDetails extends Model
{
    use HasFactory;

    protected $fillable = [
        'foodQuantity',
        'foodNote',
        'foodAmmount',
        'orderInvoiceID',
        'foodCategoryID',
    ];

    public function orderInvoice()
    {
        return $this->belongsTo(orderInvoice::class, 'orderInvoiceID');
    }

    public function foodCategory()
    {
        return $this->belongsTo(FoodCategory::class, 'foodCategoryID');
    }
}
