<?php

namespace Database\Seeders;

use App\Models\Table;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class TableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //

        Table::create([
            'name' => 'table1',
            'number' => '1',
            'capacity' => '5',
            'created_By' => '1',
        ]);

        Table::create([
            'name' => 'table2',
            'number' => '2',
            'capacity' => '3',
            'created_By' => '1',
        ]);
    }
}
