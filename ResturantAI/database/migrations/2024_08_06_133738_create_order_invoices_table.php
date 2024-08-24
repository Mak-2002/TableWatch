<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('order_invoices', function (Blueprint $table) {
            $table->id();
            $table->tinyInteger('typePayment')->nullable();
            $table->integer('tax')->nullable();
            $table->double('totalAmmount')->nullable();
            $table->tinyInteger('status');
            $table->foreignId('waiterID')->nullable()->references('id')->on('waiters');
            $table->foreignId('reservationID')->references('id')->on('reservations');
            $table->foreignId('created_By')->nullable()->references('id')->on('employees');
            $table->softDeletes();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('order_invoices');
    }
};
