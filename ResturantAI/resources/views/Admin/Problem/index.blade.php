<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Problem Table</title>

    @include('layouts.Admin.LinkHeader')

</head>

<body>
    <div id="app">
        <div class="main-wrapper main-wrapper-1">
            <div class="navbar-bg"></div>

            @include('layouts.Admin.Header')

            <div class="main-sidebar sidebar-style-2">


                @include('layouts.Admin.Sidebar')
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <section class="section">

                    <div class="row">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-header">
                                    <h4>Problem Table</h4>
                                </div>

                                {{-- message Section --}}

                                @if (session('success_message'))
                                    <div class="alert alert-success">
                                        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                            <span aria-hidden="true">&times;</span>
                                        </button>
                                        {{ session('success_message') }}
                                    </div>
                                @endif

                                @if (session('error_message'))
                                    <div class="alert alert-danger">
                                        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                            <span aria-hidden="true">&times;</span>
                                        </button>
                                        {{ session('error_message') }}
                                    </div>
                                @endif
                                {{-- end  message Section --}}
                                
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="table-1">
                                            <thead>
                                                <tr>
                                                    <th class="text-center">
                                                        #
                                                    </th>
                                                    <th>Problem Name</th>
                                                    <th>Problem Description</th>                           
                                                    <th>Table Name</th>
                                                    <th>Table Number</th>
                                                    <th>Table Capacity</th>
                                                    <th>Video Clip</th>
                                                    <th>Created Date</th>
               
                                                </tr>
                                            </thead>

                                            <tbody>
                                                @foreach ($problems as $problem)
                                                    <tr>
                                                        <td>
                                                            {{ $problem->id }}
                                                        </td>
                                                        <td>{{ $problem->problem_name }}</td>
                                                        <td>{{ $problem->problem_description}}</td>
                                                        <td>{{ $problem->table->name ?? '-' }}</td>
                                                        <td>{{ $problem->table->number ?? '-' }}</td>
                                                        <td>{{ $problem->table->capacity ?? '-' }}</td>
                                                
                                                        <td>
                                                            <video width="100" height="100" controls>
                                                                <source src="{{ asset('video/' . $problem->video_clip_path) }}" type="video/mp4">
                                                                Your browser does not support the video tag.
                                                            </video>
                                                        </td>
                                                        
                                                        <td>{{ $problem->created_at ?? '-' }}</td>
                    
                                                    </tr>
                                                @endforeach

                                            </tbody>

                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </section>
            </div>

        </div>
    </div>

    @include('layouts.Admin.LinkJS')
</body>

</html>
