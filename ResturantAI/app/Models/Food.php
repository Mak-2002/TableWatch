<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Food extends Model
{
    use HasFactory , SoftDeletes;

    protected $fillable = [
        'name',
        'details',
        'note',
        'status',
        'img',
        'created_By',
    ];


    public function employee()
    {
        return $this->belongsTo(Employee::class, 'created_By');
    }

    public function category()
    {
        return $this->hasMany(FoodCategory::class, 'categoryID');
    }
}
