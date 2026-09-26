package com.productdiscovery.web;

import com.productdiscovery.service.SearchResponse;
import com.productdiscovery.service.SearchService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.validation.annotation.Validated;
import java.io.IOException;

@RestController
@RequestMapping("/api/search")
@Validated
public class SearchController {
    private final SearchService searches;
    public SearchController(SearchService searches) { this.searches = searches; }

    @PostMapping("/text")
    public SearchResponse text(@Valid @RequestBody TextSearchRequest request) {
        return searches.searchText(request.userId(), request.query(), request.topK(), request.personalized());
    }

    @PostMapping(value = "/image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public SearchResponse image(@RequestParam Long userId,
                                @RequestParam(defaultValue = "10") @Min(1) @Max(50) int topK,
                                @RequestParam(defaultValue = "true") boolean personalized,
                                @RequestPart("file") MultipartFile file) throws IOException {
        if (file.isEmpty()) throw new IllegalArgumentException("Image must not be empty");
        return searches.searchImage(userId, file.getBytes(), file.getContentType(),
                file.getOriginalFilename(), topK, personalized);
    }

    public record TextSearchRequest(
            @NotNull Long userId,
            @NotBlank @Size(max = 1000) String query,
            @Min(1) @Max(50) int topK,
            boolean personalized
    ) {
        public TextSearchRequest { if (topK == 0) topK = 10; }
    }
}
