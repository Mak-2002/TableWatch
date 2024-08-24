<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Problem extends Model
{
    use HasFactory;

    protected $fillable = [
        'problem_name',
        'problem_description',
        'video_clip_path',
        'table_number',
    ];

    public function table()
    {
        return $this->belongsTo(Table::class, 'table_number');
    }
}
