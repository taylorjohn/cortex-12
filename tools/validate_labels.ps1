# tools/validate_labels.ps1
$labels = Get-Content "data/curriculum/labels.json" | ConvertFrom-Json
$mismatches = 0

Get-ChildItem "data/curriculum/images/*.png" | ForEach-Object {
    $filename = $_.Name
    
    # Extract expected orientation from filename (e.g., "90deg" → "90")
    if ($filename -match '_([0-9]+)deg_') {
        $expected_orient = $matches[1]
        $actual_orient = $labels.$filename.orientation
        
        if ($actual_orient -ne $expected_orient) {
            Write-Host "MISMATCH: $filename → expected=$expected_orient, got=$actual_orient" -ForegroundColor Red
            $mismatches++
        }
    }
}

if ($mismatches -eq 0) {
    Write-Host "[check] All orientation labels are CORRECT!" -ForegroundColor Green
} else {
    Write-Host "[bad] Found $mismatches label mismatches" -ForegroundColor Red
}