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
        Schema::create('order_invoice_details', function (Blueprint $table) {
            $table->id();
            $table->integer('foodQuantity');
            $table->string('foodNote');
            $table->integer('foodAmmount');
            $table->foreignId('orderInvoiceID')->nullable()->references('id')->on('order_invoices');
            $table->foreignId('foodCategoryID')->nullable()->references('id')->on('food_categories');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('order_invoice_details');
    }
};
