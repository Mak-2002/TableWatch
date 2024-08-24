<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Waiter extends Model
{
    use HasFactory , SoftDeletes;

    protected $fillable = [
        'name',
        'email',
        'gender',
        'status',
        'age',
        'phone',
        'img',
        'facePath',
        'created_By',
    ];

    public function employee()
    {
        return $this->belongsTo(Employee::class, 'created_By');
    }
}
