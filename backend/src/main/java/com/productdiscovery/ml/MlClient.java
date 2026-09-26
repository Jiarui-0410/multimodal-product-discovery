package com.productdiscovery.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.productdiscovery.config.MlProperties;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import java.util.List;

@Component
public class MlClient {
    private final RestClient client;

    public MlClient(RestClient.Builder builder, MlProperties properties) {
        this.client = builder.baseUrl(properties.baseUrl()).build();
    }

    public List<RetrievalHit> retrieveText(String query, int topK) {
        RetrievalResponse response = client.post()
                .uri("/retrieve/text")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new TextRetrievalRequest(query, topK))
                .retrieve()
                .body(RetrievalResponse.class);
        return response == null || response.results() == null ? List.of() : response.results();
    }

    public List<RetrievalHit> retrieveImage(byte[] image, String contentType, String filename, int topK) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("top_k", Integer.toString(topK));
        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentType(contentType == null
                ? MediaType.IMAGE_JPEG : MediaType.parseMediaType(contentType));
        body.add("file", new HttpEntity<>(new NamedByteArrayResource(image, filename), fileHeaders));
        RetrievalResponse response = client.post()
                .uri("/retrieve/image")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(RetrievalResponse.class);
        return response == null || response.results() == null ? List.of() : response.results();
    }

    private record TextRetrievalRequest(String query, @JsonProperty("top_k") int topK) {}
    private record RetrievalResponse(List<RetrievalHit> results) {}
    public record RetrievalHit(
            @JsonProperty("index_id") long indexId,
            double score,
            List<Double> embedding
    ) {}

    private static final class NamedByteArrayResource extends ByteArrayResource {
        private final String filename;
        private NamedByteArrayResource(byte[] bytes, String filename) {
            super(bytes);
            this.filename = filename == null ? "query-image.jpg" : filename;
        }
        @Override public String getFilename() { return filename; }
    }
}
