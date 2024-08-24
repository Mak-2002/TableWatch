<?php

namespace Database\Seeders;

use App\Models\Category;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class CategorySeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //
        Category::create([
            'name' => 'cat1',
            'type' => '1',
            'created_By' => '1'
        ]);

        Category::create([
            'name' => 'cat2',
            'type' => '0',
            'created_By' => '1'
        ]);
    }
}
