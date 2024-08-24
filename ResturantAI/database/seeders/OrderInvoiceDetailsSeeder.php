<?php

namespace Database\Seeders;

use App\Models\orderInvoiceDetails;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class OrderInvoiceDetailsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //
        orderInvoiceDetails::create([
            'foodQuantity' => '2',
            'foodNote' => 'note1',
            'foodAmmount' => '20000',
            'orderInvoiceID' => '1',
            'foodCategoryID' => '1',
        ]);

        orderInvoiceDetails::create([
            'foodQuantity' => '3',
            'foodNote' => 'note2',
            'foodAmmount' => '15000',
            'orderInvoiceID' => '1',
            'foodCategoryID' => '2',
        ]);
    }
}
