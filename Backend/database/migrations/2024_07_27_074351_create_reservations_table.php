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
        Schema::create('reservations', function (Blueprint $table) {
            $table->id();
            $table->string('day');
            $table->string('date');
            $table->time('timeStart');
            $table->time('timeEnd');
            $table->tinyInteger('status')->default(0);
            $table->foreignId('created_By')->nullable()->references('id')->on('employees');
            $table->foreignId('tableID')->references('id')->on('tables')->cascadeOnDelete();
            $table->foreignId('userID')->references('id')->on('user_r_s');
            $table->softDeletes();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('reservations');
    }
};
