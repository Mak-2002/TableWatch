<?php

namespace Database\Seeders;

use App\Models\Reservation;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class ReservationSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //

        Reservation::create([

            'day' => 'Wednesday',
            'date' => '2024-08-07',
            'timeStart' => '01:01:00',
            'timeEnd' => '02:02:00',
            'status' => '0',
            'created_By' => '1',
            'tableID' => '1',
            'userID' => '1'
        ]);

        Reservation::create([
           'day' => 'Thursday',
           'date' => '2024-08-08',
           'timeStart' => '01:01:00',
           'timeEnd' => '02:01:00',
           'status' => '0',
           'created_By' => '1',
           'tableID' => '2',
           'userID' => '2',
        ]);
    }
}
