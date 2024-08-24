<?php

namespace Database\Seeders;

use App\Models\orderInvoice;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class OrderInvoiceSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //

        orderInvoice::create([
            'status' => '0',
            'reservationID' => '1',
            'created_By' => '1',
        ]);
    }
}
