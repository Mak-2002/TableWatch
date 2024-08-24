<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class FoodCategory extends Model
{
    use HasFactory , SoftDeletes;

    protected $fillable = [
        'price',
        'note',
        'foodID',
        'categoryID',
    ];

    public function food()
    {
        return $this->belongsTo(Food::class, 'foodID');
    }

    public function category()
    {
        return $this->belongsTo(Category::class, 'categoryID');
    }

    public function orderInvoiceDetails()
    {
        return $this->hasMany(orderInvoiceDetails::class, 'foodCategoryID');
    }
}
