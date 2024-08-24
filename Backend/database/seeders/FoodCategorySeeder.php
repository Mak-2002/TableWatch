<?php

namespace Database\Seeders;

use App\Models\FoodCategory;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class FoodCategorySeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //

        FoodCategory::create([
            'price' => '10000',
            'note' => 'note1',
            'foodID' => '1',
            'categoryID' => '1',
        ]);

        FoodCategory::create([
            'price' => '5000',
            'note' => 'note2',
            'foodID' => '2',
            'categoryID' => '2',
        ]);
    }
}
