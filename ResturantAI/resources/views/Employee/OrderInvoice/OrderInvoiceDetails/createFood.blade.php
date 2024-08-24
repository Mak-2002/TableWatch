<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Food Order Invoice Details</title>

    @include('layouts.Employee.LinkHeader')

    <script>
        function calculateAmount() {
            // Get selected food category's price
            var foodCategory = document.getElementById('foodCategory');
            var price = parseFloat(foodCategory.options[foodCategory.selectedIndex].getAttribute('data-price'));

            // Get quantity entered by user
            var quantity = parseInt(document.getElementById('foodQuantity').value);

            // Calculate amount
            var amount = price * quantity;

            // Update the amount input field
            document.getElementById('foodAmount').value = amount.toFixed(2); // Adjust decimal places as needed
        }
    </script>
</head>

<body>
    <div id="app">
        <div class="main-wrapper main-wrapper-1">
            <div class="navbar-bg"></div>

            @include('layouts.Employee.Header')

            <div class="main-sidebar sidebar-style-2">
                @include('layouts.Employee.Sidebar')
            </div>
            <!-- Main Content -->
            <div class="main-content">
                <section class="section">
                    <div class="col-10 col-md-6 col-lg-12">
                        <div class="card">
                            <form method="POST" action="{{ route('employee.order.invoice.store.food') }}" enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Food Order Invoice Details</h4>
                                </div>
                                <div class="card-body">
                                    <div class="form-group">
                                        <div class="row">
                                            <div class="col-5">
                                                <label>Food Categories</label>
                                                <select id="foodCategory" class="form-control select2" name="foodCategoryID" required onchange="calculateAmount()">
                                                    @foreach ($foodCategories as $foodCategory)
                                                        <option value="{{ $foodCategory->id }}" data-price="{{ $foodCategory->price }}">
                                                            {{ $foodCategory->food->name }} - {{ $foodCategory->food->details }} - {{ $foodCategory->price }}SYP - {{ $foodCategory->category->name }} -
                                                            @if ($foodCategory->category->type == 0)
                                                                <div class="badge badge-success">Meal</div>
                                                            @elseif($foodCategory->category->type == 1)
                                                                <div class="badge badge-success">Sandwich</div>
                                                            @endif
                                                        </option>
                                                    @endforeach
                                                </select>
                                            </div>
                                        </div>
                                        <br>
                                        <div class="row">
                                            <div class="col">
                                                <label>Quantity</label>
                                                <input id="foodQuantity" type="number" class="form-control" name="foodQuantity" required>
                                            </div>
                                            <div class="col">
                                                <label>Note</label>
                                                <input type="text" class="form-control" name="foodNote" required>
                                            </div>
                                            <div class="col">
                                                <label>Amount</label>
                                                <input id="foodAmount" type="number" class="form-control" name="foodAmmount" required readonly>
                                            </div>
                                            <input type="hidden" class="form-control" name="orderInvoiceID" required value="{{ $orderInvoiceID }}">
                                        </div>
                                    </div>
                                </div>
                                <div class="card-footer text-center">
                                    <button class="btn btn-primary" type="submit">Add</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>
    @include('layouts.Employee.LinkJS')
</body>

</html>
