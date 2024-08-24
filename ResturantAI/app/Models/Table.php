<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Table extends Model
{
    use HasFactory , SoftDeletes;

    protected $fillable = [
        'name',
        'number',
        'capacity',
        'created_By',
    ];

    public function reservation()
    {
        return $this->belongsTo(Reservation::class, 'tableID');
    }

    public function employee()
    {
        return $this->belongsTo(Employee::class, 'created_By');
    }
}
