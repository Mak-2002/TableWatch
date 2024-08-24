<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Reservation extends Model
{
    use HasFactory, SoftDeletes;

    protected $fillable = [
        'day',
        'date',
        'timeStart',
        'timeEnd',
        'status',
        'tableID',
        'userID',
        'created_By',
    ];

    public function employee()
    {
        return $this->belongsTo(Employee::class, 'created_By');
    }

    public function table()
    {
        return $this->belongsTo(Table::class, 'tableID');
    }

    public function user()
    {
        return $this->belongsTo(UserR::class, 'userID');
    }
}
