<?php

namespace Database\Seeders;

use App\Models\Food;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class FoodSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //
        Food::create([
            'name' => 'food1',
            'details' => 'No details',
            'note' => 'note1',
            'status' => '1',
            'created_By' => '1',
        ]);

        Food::create([
            'name' => 'food2',
            'details' => 'No details',
            'note' => 'note2',
            'status' => '0',
            'created_By' => '1',
        ]);
    }
}
